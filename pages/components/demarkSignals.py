import pandas as pd
import numpy as np


# ================================================================
# TD SEQUENTIAL SETUP (1–9)
# ================================================================
def calculate_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    TD Sequential Setup counting:
      - Buy Setup: close < close 4 bars earlier
      - Sell Setup: close > close 4 bars earlier
      - Completes at count 9
    """
    df["Buy Setup"] = np.nan
    df["Sell Setup"] = np.nan

    close_vals = df["Close"].values
    buy_count = np.zeros(len(df), dtype=np.int32)
    sell_count = np.zeros(len(df), dtype=np.int32)

    for i in range(len(df)):
        if i < 4:
            continue

        is_buy = close_vals[i] < close_vals[i - 4]
        is_sell = close_vals[i] > close_vals[i - 4]

        # update counts
        if is_buy:
            buy_count[i] = buy_count[i - 1] + 1
            sell_count[i] = 0
        else:
            buy_count[i] = 0

        if is_sell:
            sell_count[i] = sell_count[i - 1] + 1
            buy_count[i] = 0
        else:
            sell_count[i] = 0

        # label
        if buy_count[i] == 9:
            df.at[df.index[i], "Buy Setup"] = "Buy Setup Completed"
            buy_count[i] = 0
        elif buy_count[i] > 0:
            df.at[df.index[i], "Buy Setup"] = f"Buy Setup {buy_count[i]}"

        if sell_count[i] == 9:
            df.at[df.index[i], "Sell Setup"] = "Sell Setup Completed"
            sell_count[i] = 0
        elif sell_count[i] > 0:
            df.at[df.index[i], "Sell Setup"] = f"Sell Setup {sell_count[i]}"

    return df


# ================================================================
# TD COUNTDOWN (1–13)
# ================================================================
def calculate_td_countdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    After Setup Completion:
      - Buy Countdown: close < close 2 bars earlier
      - Sell Countdown: close > close 2 bars earlier
      - Completes at 13
    """
    df["Buy Countdown"] = np.nan
    df["Sell Countdown"] = np.nan

    close_vals = df["Close"].values
    buy_cd = np.zeros(len(df), dtype=np.int32)
    sell_cd = np.zeros(len(df), dtype=np.int32)

    for i in range(len(df)):
        if i < 2:
            continue

        # Start countdown at setup completion
        if df.at[df.index[i], "Buy Setup"] == "Buy Setup Completed":
            buy_cd[i] = 1

        if df.at[df.index[i], "Sell Setup"] == "Sell Setup Completed":
            sell_cd[i] = 1

        # Continue Buy Countdown
        if buy_cd[i - 1] > 0 and close_vals[i] < close_vals[i - 2]:
            buy_cd[i] = buy_cd[i - 1] + 1
            df.at[df.index[i], "Buy Countdown"] = f"Buy Countdown {buy_cd[i]}"
            if buy_cd[i] == 13:
                df.at[df.index[i], "Buy Countdown"] = "Buy Countdown Completed"

        # Continue Sell Countdown
        if sell_cd[i - 1] > 0 and close_vals[i] > close_vals[i - 2]:
            sell_cd[i] = sell_cd[i - 1] + 1
            df.at[df.index[i], "Sell Countdown"] = f"Sell Countdown {sell_cd[i]}"
            if sell_cd[i] == 13:
                df.at[df.index[i], "Sell Countdown"] = "Sell Countdown Completed"

    return df


# ================================================================
# TD DEMAND & SUPPLY LINES (F% SPACE)
# ================================================================
def calculate_td_demand_supply_lines_fpercent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ringed lows → TD Demand Line F
    Ringed highs → TD Supply Line F
    Forward-filled to create continuous lines.
    """
    df["TD Demand Line F"] = np.nan
    df["TD Supply Line F"] = np.nan

    f_vals = df["F_numeric"].to_numpy()
    demand_points = []
    supply_points = []

    for i in range(1, len(df) - 1):
        # Ringed Low
        if f_vals[i] < f_vals[i - 1] and f_vals[i] < f_vals[i + 1]:
            demand_points.append(f_vals[i])
            if len(demand_points) >= 2:
                df.at[df.index[i], "TD Demand Line F"] = max(demand_points[-2:])
            else:
                df.at[df.index[i], "TD Demand Line F"] = demand_points[-1]

        # Ringed High
        if f_vals[i] > f_vals[i - 1] and f_vals[i] > f_vals[i + 1]:
            supply_points.append(f_vals[i])
            if len(supply_points) >= 2:
                df.at[df.index[i], "TD Supply Line F"] = min(supply_points[-2:])
            else:
                df.at[df.index[i], "TD Supply Line F"] = supply_points[-1]

    df["TD Demand Line F"] = df["TD Demand Line F"].ffill()
    df["TD Supply Line F"] = df["TD Supply Line F"].ffill()
    return df


# ================================================================
# TD SUPPLY CROSS ALERTS (F%)
# ================================================================
def calculate_td_supply_cross_alert(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects F_numeric crossing above or below the TD Supply Line F.
    """
    df["tdSupplyCrossalert"] = ""

    if "TD Supply Line F" not in df.columns:
        return df

    for i in range(1, len(df)):
        prev_f = df["F_numeric"].iloc[i - 1]
        curr_f = df["F_numeric"].iloc[i]

        prev_sup = df["TD Supply Line F"].iloc[i - 1]
        curr_sup = df["TD Supply Line F"].iloc[i]

        if prev_f < prev_sup and curr_f >= curr_sup:
            df.at[df.index[i], "tdSupplyCrossalert"] = "cross"

        elif prev_f > prev_sup and curr_f <= curr_sup:
            df.at[df.index[i], "tdSupplyCrossalert"] = "down"

    return df


# ================================================================
# CLEAN TDST
# ================================================================
def calculate_clean_tdst(df: pd.DataFrame) -> pd.DataFrame:
    """
    TDST based on Setup Completion:
      - Buy TDST = max(Setup bars 1 and 2 highs)
      - Sell TDST = low of bar 1 of Sell Setup
    """
    df["TDST"] = None
    current_tdst = None

    for i in range(9, len(df)):
        # Buy Setup Completed
        if df["Buy Setup"].iloc[i] == "Buy Setup Completed":
            bs1_high = df["High"].iloc[i - 8]
            bs2_high = df["High"].iloc[i - 7]
            current_tdst = f"Buy TDST: {round(max(bs1_high, bs2_high), 2)}"
            df.at[df.index[i], "TDST"] = current_tdst

        # Sell Setup Completed
        elif df["Sell Setup"].iloc[i] == "Sell Setup Completed":
            ss1_low = df["Low"].iloc[i - 8]
            current_tdst = f"Sell TDST: {round(ss1_low, 2)}"
            df.at[df.index[i], "TDST"] = current_tdst

        else:
            df.at[df.index[i], "TDST"] = None

    return df


# ================================================================
# HEAVEN CLOUD ☁️
# ================================================================
def calculate_heaven_cloud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cloud appears when F_numeric is above the TD Supply Line F.
    """
    if "TD Supply Line F" not in df.columns:
        df["Heaven_Cloud"] = ""
        return df

    df["Heaven_Cloud"] = np.where(
        df["F_numeric"] > df["TD Supply Line F"],
        "☁️",
        ""
    )
    return df


# ================================================================
# DRIZZLE EMOJI 🌧️ (Demand breakdown)
# ================================================================
def calculate_drizzle_emoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drizzle when price crosses down below TD Demand Line F.
    """
    if "TD Demand Line F" not in df.columns:
        df["Drizzle_Emoji"] = ""
        return df

    df["Prev_F"] = df["F_numeric"].shift(1)
    df["Prev_Demand"] = df["TD Demand Line F"].shift(1)

    df["Drizzle_Emoji"] = np.where(
        (df["Prev_F"] >= df["Prev_Demand"]) &
        (df["F_numeric"] < df["TD Demand Line F"]),
        "🌧️",
        ""
    )
    return df
