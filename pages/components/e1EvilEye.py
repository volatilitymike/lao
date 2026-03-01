import pandas as pd
import numpy as np

def apply_e1_kijun_evil_eye(df: pd.DataFrame) -> pd.DataFrame:
    """
    🧿 Evil Eye at Entry 1:
    If Entry 1 (🎯) happens on a bar where Mike crosses Kijun_F
    (between previous bar and this bar), mark 🧿 in its own column
    WITHOUT changing the original E1 emojis.

    New column: E1_EvilEye_Emoji
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    needed = ["F_numeric", "Kijun_F",
              "Call_FirstEntry_Emoji", "Put_FirstEntry_Emoji"]
    if any(col not in out.columns for col in needed):
        return out

    # new, independent layer
    out["E1_EvilEye_Emoji"] = ""

    for idx in out.index:
        is_call_e1 = out.at[idx, "Call_FirstEntry_Emoji"] == "🎯"
        is_put_e1  = out.at[idx, "Put_FirstEntry_Emoji"]  == "🎯"

        if not (is_call_e1 or is_put_e1):
            continue

        pos = out.index.get_loc(idx)
        if pos == 0:
            # no previous bar to compare
            continue

        prev_idx = out.index[pos - 1]

        curr_mike = out.at[idx,      "F_numeric"]
        curr_kij  = out.at[idx,      "Kijun_F"]
        prev_mike = out.at[prev_idx, "F_numeric"]
        prev_kij  = out.at[prev_idx, "Kijun_F"]

        if any(pd.isna(v) for v in [curr_mike, curr_kij, prev_mike, prev_kij]):
            continue

        curr_diff = curr_mike - curr_kij
        prev_diff = prev_mike - prev_kij

        eps = 1e-6
        if abs(curr_diff) < eps:
            curr_diff = 0.0
        if abs(prev_diff) < eps:
            prev_diff = 0.0

        crossed = (curr_diff == 0) or (prev_diff == 0) or (curr_diff * prev_diff < 0)

        if crossed:
            out.at[idx, "E1_EvilEye_Emoji"] = "🧿"

    return out
