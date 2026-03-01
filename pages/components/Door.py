import pandas as pd
import numpy as np

def apply_T0_door(
    df: pd.DataFrame,
    band_distance: float = 5
) -> pd.DataFrame:
    """
    Attach T0 Door (🚪) when F_numeric gets close to either:
        - Upper Bollinger Band (CALL path)
        - Lower Bollinger Band (PUT path)

    BUT ONLY if:
        - A First Entry (🎯) exists,
        - And the door happens *after* that first Entry 1 in time.

    If both Put and Call have 🎯, we follow whichever Entry 1 comes first.
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    # Always reset T0 cleanly
    df["T0_Emoji"] = ""

    # Must have these columns
    needed = ["F_numeric", "F% Upper", "F% Lower"]
    if not all(x in df.columns for x in needed):
        return df

    f = pd.to_numeric(df["F_numeric"], errors="coerce")
    upper = pd.to_numeric(df["F% Upper"], errors="coerce")
    lower = pd.to_numeric(df["F% Lower"], errors="coerce")

    # Find first Entry 1 on each side
    put_E1 = df.index[df.get("Put_FirstEntry_Emoji", "") == "🎯"]
    call_E1 = df.index[df.get("Call_FirstEntry_Emoji", "") == "🎯"]

    # Decide which side & where to start
    side = None      # "put" or "call"
    start_loc = None

    if len(put_E1) > 0:
        side = "put"
        start_loc = df.index.get_loc(put_E1[0])

    if len(call_E1) > 0:
        call_loc = df.index.get_loc(call_E1[0])
        if side is None or call_loc < start_loc:
            side = "call"
            start_loc = call_loc

    # No Entry 1 at all → no door
    if side is None or start_loc is None:
        return df

    # Scan *after* the first Entry 1 bar
    if side == "put":
        for i in range(start_loc + 1, len(df)):
            if (
                pd.notna(f.iloc[i])
                and pd.notna(lower.iloc[i])
                and abs(f.iloc[i] - lower.iloc[i]) <= band_distance
            ):
                df.at[df.index[i], "T0_Emoji"] = "🚪"
                break

    elif side == "call":
        for i in range(start_loc + 1, len(df)):
            if (
                pd.notna(f.iloc[i])
                and pd.notna(upper.iloc[i])
                and abs(f.iloc[i] - upper.iloc[i]) <= band_distance
            ):
                df.at[df.index[i], "T0_Emoji"] = "🚪"
                break

    return df
