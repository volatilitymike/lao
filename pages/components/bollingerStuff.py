# pages/components/bollingerStuff.py

import pandas as pd
import numpy as np


def calculate_f_std_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Bollinger-style bands on Mike axis (F_numeric):

        F% MA    = rolling mean of F_numeric
        F% Std   = rolling std of F_numeric
        F% Upper = F% MA + 2 * F% Std
        F% Lower = F% MA - 2 * F% Std
    """
    if "F_numeric" not in df.columns or df.empty:
        df["F% MA"] = np.nan
        df["F% Std"] = np.nan
        df["F% Upper"] = np.nan
        df["F% Lower"] = np.nan
        return df

    df["F% MA"] = df["F_numeric"].rolling(window=window, min_periods=1).mean()
    df["F% Std"] = df["F_numeric"].rolling(window=window, min_periods=1).std()
    df["F% Upper"] = df["F% MA"] + (2 * df["F% Std"])
    df["F% Lower"] = df["F% MA"] - (2 * df["F% Std"])

    return df


def calculate_f_bbw(df: pd.DataFrame, scale_factor: float = 10.0) -> pd.DataFrame:
    """
    Computes Bollinger Band Width (BBW) for F% and scales it:

        BBW_raw = (Upper - Lower) / |Middle| * 100
        F% BBW  = BBW_raw / scale_factor

    Fills NaNs with 0.
    """
    if not {"F% Upper", "F% Lower", "F% MA"}.issubset(df.columns):
        df["F% BBW"] = 0.0
        return df

    ma_abs = df["F% MA"].abs().replace(0, np.nan)
    bbw = ((df["F% Upper"] - df["F% Lower"]) / ma_abs) * 100.0
    bbw_scaled = (bbw / scale_factor).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["F% BBW"] = bbw_scaled
    return df


def detect_bbw_tight(
    df: pd.DataFrame,
    window: int = 5,
    percentile_threshold: float = 10.0,
) -> pd.DataFrame:
    """
    Detects BBW Tight Compression using dynamic threshold based on the
    ticker's own BBW distribution.

    - dynamic_threshold = Xth percentile of non-null F% BBW
    - BBW_Tight = F% BBW < dynamic_threshold
    - BBW_Tight_Emoji = '🐝' when at least 3 of last `window` bars are tight
    """
    if "F% BBW" not in df.columns or df["F% BBW"].dropna().empty:
        df["BBW_Tight"] = False
        df["BBW_Tight_Emoji"] = ""
        return df

    dynamic_threshold = np.percentile(df["F% BBW"].dropna(), percentile_threshold)

    df["BBW_Tight"] = df["F% BBW"] < dynamic_threshold
    df["BBW_Tight_Emoji"] = ""

    for i in range(window, len(df)):
        recent = df["BBW_Tight"].iloc[i - window : i]
        if recent.sum() >= 3:
            df.at[df.index[i], "BBW_Tight_Emoji"] = "🐝"

    return df


def add_bbw_anchor_and_ratio(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Adds:
        - BBW_Anchor = F% BBW shifted by `lookback`
        - BBW_Ratio  = F% BBW / BBW_Anchor
        - BBW Alert  = '🔥' when BBW_Ratio >= 2 (double or more)
    """
    if "F% BBW" not in df.columns or df.empty:
        df["BBW_Anchor"] = np.nan
        df["BBW_Ratio"] = np.nan
        df["BBW Alert"] = ""
        return df

    df["BBW_Anchor"] = df["F% BBW"].shift(lookback)
    df["BBW_Ratio"] = df["F% BBW"] / df["BBW_Anchor"]

    def bbw_alert(row):
        if pd.isna(row["BBW_Ratio"]):
            return ""
        if row["BBW_Ratio"] >= 2:
            return "🔥"  # Double+ Expansion
        return ""

    df["BBW Alert"] = df.apply(bbw_alert, axis=1)
    return df


def calculate_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compliance = (F% BBW - BBW_Anchor) / RVOL_5

    Measures how much expansion is happening per unit of relative volume.
    """
    df["Compliance"] = np.nan
    required_cols = {"F% BBW", "BBW_Anchor", "RVOL_5"}
    if not required_cols.issubset(df.columns):
        return df

    delta_bbw = df["F% BBW"] - df["BBW_Anchor"]
    pressure = df["RVOL_5"].replace(0, np.nan)
    df["Compliance"] = (delta_bbw / pressure).replace([np.inf, -np.inf], np.nan)

    return df


def detect_compliance_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 🫧 emoji where Compliance shifts from negative to >= 0.
    """
    df["Compliance Shift"] = ""
    if "Compliance" not in df.columns or df["Compliance"].dropna().empty:
        return df

    for i in range(1, len(df)):
        prev = df["Compliance"].iloc[i - 1]
        curr = df["Compliance"].iloc[i]
        if pd.notna(prev) and pd.notna(curr):
            if prev < 0 and curr >= 0:
                df.at[df.index[i], "Compliance Shift"] = "🫧"

    return df


def detect_marengo(df: pd.DataFrame, rvol_threshold: float = 1.2) -> pd.DataFrame:
    """
    Detects North & South Marengo:

        - North Marengo (Marengo column):
            Mike (F_numeric) >= F% Upper AND RVOL_5 > rvol_threshold

        - South Marengo (South_Marengo column):
            Mike (F_numeric) <= F% Lower AND RVOL_5 > rvol_threshold
    """
    if not {"F_numeric", "F% Upper", "F% Lower", "RVOL_5"}.issubset(df.columns):
        df["Marengo"] = ""
        df["South_Marengo"] = ""
        return df

    df["Marengo"] = ""
    df["South_Marengo"] = ""

    for i in range(len(df)):
        mike = df.at[df.index[i], "F_numeric"]
        upper = df.at[df.index[i], "F% Upper"]
        lower = df.at[df.index[i], "F% Lower"]
        rvol = df.at[df.index[i], "RVOL_5"]

        if pd.notna(mike) and pd.notna(upper) and pd.notna(lower) and pd.notna(rvol):
            if mike >= upper and rvol > rvol_threshold:
                df.at[df.index[i], "Marengo"] = "🐎"
            elif mike <= lower and rvol > rvol_threshold:
                df.at[df.index[i], "South_Marengo"] = "🐎"

    return df


def apply_bollinger_suite(
    df: pd.DataFrame,
    window: int = 20,
    scale_factor: float = 10.0,
    tight_window: int = 5,
    percentile_threshold: float = 10.0,
    anchor_lookback: int = 5,
    rvol_threshold: float = 1.2,
) -> pd.DataFrame:
    """
    Convenience wrapper to run the full F-space Bollinger stack in one call:
        - F% bands
        - F% BBW
        - BBW tight 🐝
        - BBW anchor / ratio / 🔥 alert
        - Compliance + 🫧 shifts
        - Marengo (🐎 north/south)
    """
    df = calculate_f_std_bands(df, window=window)
    df = calculate_f_bbw(df, scale_factor=scale_factor)
    df = detect_bbw_tight(df, window=tight_window, percentile_threshold=percentile_threshold)
    df = add_bbw_anchor_and_ratio(df, lookback=anchor_lookback)
    df = calculate_compliance(df)
    df = detect_compliance_shift(df)
    df = detect_marengo(df, rvol_threshold=rvol_threshold)
    return df
