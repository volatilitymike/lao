# pages/components/stdExpansion.py

import pandas as pd
import numpy as np


def apply_std_expansion(
    df: pd.DataFrame,
    window: int = 9,
    anchor_lookback: int = 5,
) -> pd.DataFrame:
    """
    F% STD Expansion on Mike axis (F_numeric):

    - F%_STD      = rolling std of F_numeric over `window`
    - STD_Anchor  = F%_STD shifted by `anchor_lookback` bars
    - STD_Ratio   = F%_STD / STD_Anchor
    - STD_Alert   = '🐦‍🔥' when STD_Ratio >= 2 (double or triple expansion)
    """

    if "F_numeric" not in df.columns or df.empty:
        df["F%_STD"] = np.nan
        df["STD_Anchor"] = np.nan
        df["STD_Ratio"] = np.nan
        df["STD_Alert"] = ""
        return df

    # 1) rolling std of F_numeric
    df["F%_STD"] = df["F_numeric"].rolling(window=window, min_periods=1).std()

    # 2) anchor and ratio
    df["STD_Anchor"] = df["F%_STD"].shift(anchor_lookback)
    df["STD_Ratio"] = df["F%_STD"] / df["STD_Anchor"]

    # 3) emoji alerts (double or triple both → 🐦‍🔥)
    df["STD_Alert"] = ""
    valid = df["STD_Ratio"].notna()
    mask_double = valid & (df["STD_Ratio"] >= 2)

    df.loc[mask_double, "STD_Alert"] = "🐦‍🔥"

    return df
