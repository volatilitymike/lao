import pandas as pd
import numpy as np

def apply_T2_lightning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks T2: momentum confirmation after T1.
    ⚡ appears on the first bar after T1 where Close extends in the
    same direction as Entry 1 (up for calls, down for puts).
    """
    out = df.copy()

    # Ensure dependencies exist
    need_cols = ["Close", "T1_Emoji", "Call_FirstEntry_Emoji", "Put_FirstEntry_Emoji"]
    for col in need_cols:
        if col not in out.columns:
            out["T2_Emoji"] = ""
            return out

    # Find Entry 1
    call_idx = out.index[out["Call_FirstEntry_Emoji"] == "🎯"]
    put_idx  = out.index[out["Put_FirstEntry_Emoji"]  == "🎯"]

    if len(call_idx) == 0 and len(put_idx) == 0:
        out["T2_Emoji"] = ""
        return out

    # Get first entry index (earliest between call/put)
    first_call_i = out.index.get_loc(call_idx[0]) if len(call_idx) > 0 else None
    first_put_i  = out.index.get_loc(put_idx[0])  if len(put_idx)  > 0 else None
    start_i = min(i for i in [first_call_i, first_put_i] if i is not None)

    # Find T1
    t1_idx = out.index[out["T1_Emoji"] == "🏇🏼"]
    if len(t1_idx) == 0:
        out["T2_Emoji"] = ""
        return out
    i_t1 = out.index.get_loc(t1_idx[0])

    # Init column
    out["T2_Emoji"] = ""

    # Loop after T1
    for i in range(i_t1 + 1, len(out)):
        prev_close = out.at[out.index[i - 1], "Close"]
        curr_close = out.at[out.index[i], "Close"]

        if pd.isna(prev_close) or pd.isna(curr_close):
            continue

        # Call case (northbound)
        if first_call_i is not None and first_call_i <= start_i:
            if curr_close > prev_close:
                out.at[out.index[i], "T2_Emoji"] = "⚡"
                break

        # Put case (southbound)
        if first_put_i is not None and first_put_i <= start_i:
            if curr_close < prev_close:
                out.at[out.index[i], "T2_Emoji"] = "⚡"
                break

    return out
