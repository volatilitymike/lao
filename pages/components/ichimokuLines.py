import pandas as pd
import numpy as np


def calculate_kijun_sen(df: pd.DataFrame, period: int = 26) -> pd.DataFrame:
    """
    Compute Kijun-sen (Base Line) in price space:
        Kijun_sen = (Highest High + Lowest Low) / 2 over `period` bars.
    """
    if df.empty or "High" not in df.columns or "Low" not in df.columns:
        df["Kijun_sen"] = np.nan
        return df

    highest_high = df["High"].rolling(window=period, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()

    df["Kijun_sen"] = (highest_high + lowest_low) / 2
    return df


def calculate_tenkan_sen(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """
    Compute Tenkan-sen (Conversion Line) in price space:
        Tenkan_sen = (Highest High + Lowest Low) / 2 over `period` bars.
    """
    if df.empty or "High" not in df.columns or "Low" not in df.columns:
        df["Tenkan_sen"] = np.nan
        return df

    highest_high = df["High"].rolling(window=period, min_periods=1).max()
    lowest_low = df["Low"].rolling(window=period, min_periods=1).min()

    df["Tenkan_sen"] = (highest_high + lowest_low) / 2
    return df


def apply_ichimoku_f_levels(
    df: pd.DataFrame,
    prev_close: float,
    tenkan_period: int = 9,
    kijun_period: int = 26,
) -> pd.DataFrame:
    """
    Apply Ichimoku core lines in Mike-space (F axis), using the *previous daily close*.

    Adds:
        - Kijun_sen  (price)
        - Kijun_F    (F-level)
        - Tenkan_sen (price)
        - F% Tenkan  (F-level)
        - Tenkan_Kijun_Cross: "🦅" (bullish) / "🐦‍⬛" (bearish) / "" (no cross)
    """
    if df.empty:
        df["Kijun_sen"] = np.nan
        df["Tenkan_sen"] = np.nan
        df["Kijun_F"] = 0
        df["F% Tenkan"] = 0
        df["Tenkan_Kijun_Cross"] = ""
        return df

    # 1) price-space lines
    df = calculate_kijun_sen(df, period=kijun_period)
    df = calculate_tenkan_sen(df, period=tenkan_period)

    # 2) F-space projections (Mike-space)
    if prev_close is None or prev_close == 0:
        df["Kijun_F"] = 0
        df["F% Tenkan"] = 0
    else:
        kijun_f = ((df["Kijun_sen"] - prev_close) / prev_close) * 10000
        kijun_f = kijun_f.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["Kijun_F"] = kijun_f.round(0).astype(int)

        tenkan_f = ((df["Tenkan_sen"] - prev_close) / prev_close) * 10000
        tenkan_f = tenkan_f.replace([np.inf, -np.inf], np.nan).fillna(0)
        df["F% Tenkan"] = tenkan_f.round(0).astype(int)

    # 3) Tenkan–Kijun cross logic (price-space)
    df["Tenkan_Kijun_Cross"] = ""

    try:
        diff = df["Tenkan_sen"] - df["Kijun_sen"]
        prev_diff = diff.shift(1)

        # Bullish cross: Tenkan goes from below/at to above Kijun
        bull_cross = (prev_diff <= 0) & (diff > 0)

        # Bearish cross: Tenkan goes from above/at to below Kijun
        bear_cross = (prev_diff >= 0) & (diff < 0)

        df.loc[bull_cross, "Tenkan_Kijun_Cross"] = "🦅"
        df.loc[bear_cross, "Tenkan_Kijun_Cross"] = "🐦‍⬛"
    except Exception:
        # If anything fails, just keep the column as empty strings
        pass

    return df
