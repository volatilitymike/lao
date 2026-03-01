import pandas as pd
import numpy as np

def apply_T1_horse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach T1 (🏇🏼) at the first horse signal AFTER the first Entry 1.

    Requirements:
      - Put_FirstEntry_Emoji == 🎯  OR  Call_FirstEntry_Emoji == 🎯
      - Then find first 'Marengo' or 'South_Marengo' row
      - Mark that bar with T1_Emoji = 🏇🏼
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    # Ensure column exists
    df["T1_Emoji"] = ""

    # Must have horses
    if "Marengo" not in df.columns and "South_Marengo" not in df.columns:
        return df

    # Find first Entry 1 (either side)
    put_E1 = df.index[df["Put_FirstEntry_Emoji"] == "🎯"]
    call_E1 = df.index[df["Call_FirstEntry_Emoji"] == "🎯"]

    start_idx = None

    if len(put_E1) > 0:
        start_idx = put_E1[0]

    if len(call_E1) > 0:
        if start_idx is None:
            start_idx = call_E1[0]
        else:
            # choose earlier of the two
            call_i = call_E1[0]
            if df.index.get_loc(call_i) < df.index.get_loc(start_idx):
                start_idx = call_i

    # No Entry 1 found → no T1
    if start_idx is None:
        return df

    start_loc = df.index.get_loc(start_idx)

    # Scan forward for first horse
    fwd = df.iloc[start_loc + 1 :]

    horse_mask = (fwd.get("Marengo") == "🐎") | (fwd.get("South_Marengo") == "🐎")

    hits = fwd[horse_mask]

    if hits.empty:
        return df

    # Mark only the first
    t1_row = hits.iloc[0]
    df.at[t1_row.name, "T1_Emoji"] = "🏇🏼"

    return df
