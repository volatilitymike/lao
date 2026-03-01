import pandas as pd
import numpy as np

def compute_midas_curves(df, price_col="F_numeric", volume_col="Volume"):
    """
    Computes MIDAS_Bear (anchor = max) and MIDAS_Bull (anchor = min)
    for any intraday DataFrame.
    """
    df = df.copy()

    # Ensure clean numeric series
    prices = df[price_col].astype(float).to_numpy()
    volume = df[volume_col].astype(float).to_numpy()

    # ---------- BEARISH (anchor = highest price) ----------
    idx_bear = np.argmax(prices)
    midas_bear = [np.nan] * len(df)

    for i in range(idx_bear, len(df)):
        window_p = prices[idx_bear:i+1]
        window_v = volume[idx_bear:i+1]
        w = window_v / window_v.sum()
        midas_bear[i] = np.sum(window_p * w)

    df["MIDAS_Bear"] = midas_bear

    # ---------- BULLISH (anchor = lowest price) ----------
    idx_bull = np.argmin(prices)
    midas_bull = [np.nan] * len(df)

    for i in range(idx_bull, len(df)):
        window_p = prices[idx_bull:i+1]
        window_v = volume[idx_bull:i+1]
        w = window_v / window_v.sum()
        midas_bull[i] = np.sum(window_p * w)

    df["MIDAS_Bull"] = midas_bull

    return df
