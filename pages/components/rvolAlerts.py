# pages/components/rvolAlerts.py

import pandas as pd
import numpy as np


def classify_rvol_value(rvol_value: float) -> str:
    """
    Classify a single RVOL value:

    - Extreme: RVOL > 1.8           → "🔺 Extreme"
    - Strong:  1.5 ≤ RVOL < 1.8     → "🟧 Strong"
    - Moderate:1.2 ≤ RVOL < 1.5     → "Moderate"
    - None:    RVOL < 1.2           → ""
    """
    if pd.isna(rvol_value):
        return ""
    if rvol_value > 1.8:
        return "🔺 Extreme"
    elif rvol_value >= 1.5:
        return "🟧 Strong"
    elif rvol_value >= 1.2:
        return "Moderate"
    else:
        return ""


def apply_rvol_alerts(
    df: pd.DataFrame,
    rvol_col: str = "RVOL_5",
    alert_col: str = "RVOL_Alert",
) -> pd.DataFrame:
    """
    Adds a column `alert_col` with RVOL classification labels:

    - uses `rvol_col` (default: "RVOL_5")
    - output examples: "🔺 Extreme", "🟧 Strong", "Moderate", ""

    If rvol_col is missing, column is set to empty strings.
    """
    if rvol_col not in df.columns or df.empty:
        df[alert_col] = ""
        return df

    df[alert_col] = df[rvol_col].apply(classify_rvol_value)
    return df
