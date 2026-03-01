import pandas as pd
import numpy as np  # currently only used for pd.isna; OK to keep or remove


# ================================================================
#  TD OPEN — Reversal after gap
# ================================================================
def td_open_signals(intraday_df: pd.DataFrame,
                    prev_high: float,
                    prev_low: float,
                    gap_type: str) -> pd.DataFrame:
    """
    TD Open:
    - If gap up day (gap_type == 'UP') and price trades back to / through prev_high → sell signal.
    - If gap down day (gap_type == 'DOWN') and price trades back to / through prev_low → buy signal.
    """

    def check_td_open(row: pd.Series) -> str:
        if gap_type == "UP":
            if row["Low"] <= prev_high:
                return "Sell SIGNAL (Reversed Down)"
        elif gap_type == "DOWN":
            if row["High"] >= prev_low:
                return "Buy SIGNAL (Reversed Up)"
        return ""

    intraday_df["TD Open"] = intraday_df.apply(check_td_open, axis=1)
    return intraday_df


# ================================================================
#  TD TRAP — Opening inside previous day's range
# ================================================================
def td_trap_signals(df: pd.DataFrame,
                    prev_high: float,
                    prev_low: float) -> pd.DataFrame:
    """
    TD Trap (day-level):
    - Today's OPEN must be inside yesterday's high/low.
    - First breakout bar determines:
        * High > prev_high → 'TD Trap BUY'
        * Low  < prev_low  → 'TD Trap SELL'
    - After first breakout, no further signals.
    """
    if df.empty:
        df["TD Trap"] = ""
        return df

    # 1) Today's open must be trapped inside yesterday's range
    day_open = df["Open"].iloc[0]
    if not (prev_low < day_open < prev_high):
        df["TD Trap"] = ""
        return df

    breakout_found = False
    signals: list[str] = []

    for _, row in df.iterrows():
        if breakout_found:
            signals.append("")
            continue

        # BUY trap: breakout above yesterday's high
        if row["High"] > prev_high:
            signals.append("TD Trap BUY")
            breakout_found = True

        # SELL trap: breakout below yesterday's low
        elif row["Low"] < prev_low:
            signals.append("TD Trap SELL")
            breakout_found = True

        else:
            signals.append("")

    df["TD Trap"] = signals
    return df


# ================================================================
#  TD CLoPWIN — Bar-to-bar continuation
# ================================================================
def td_clopwin_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    TD CLoPWIN:
    - Current bar's open and close are BOTH contained within prior bar's open/close range.
    - If current close > prior close  → 'Buy CLoPWIN'
    - If current close < prior close  → 'Sell CLoPWIN'
    """

    def check_clopwin(row: pd.Series) -> str:
        idx = row.name
        if idx == 0:
            return ""

        prev_row = df.iloc[idx - 1]
        prev_open = prev_row["Open"]
        prev_close = prev_row["Close"]

        # Containment inside previous bar body
        body_low = min(prev_open, prev_close)
        body_high = max(prev_open, prev_close)

        contained = (
            body_low <= row["Open"] <= body_high
            and
            body_low <= row["Close"] <= body_high
        )
        if not contained:
            return ""

        if row["Close"] > prev_close:
            return "Buy CLoPWIN"
        if row["Close"] < prev_close:
            return "Sell CLoPWIN"
        return ""

    df["TD CLoPWIN"] = df.apply(check_clopwin, axis=1)
    return df


# ================================================================
#  TD CLoP (Close-Open Logic)
# ================================================================
def td_clop_signals(intraday_df: pd.DataFrame,
                    prev_open: float,
                    prev_close: float) -> pd.DataFrame:
    """
    TD CLoP (daily-level):
    - BUY: opens below both prev_open & prev_close, then trades above both.
    - SELL: opens above both prev_open & prev_close, then trades below both.
    """

    def check_td_clop(row: pd.Series) -> str:
        # BUY: Opens below yesterday’s open & close → then rallies above both
        if (row["Open"] < prev_open and row["Open"] < prev_close
                and row["High"] > prev_open and row["High"] > prev_close):
            return "Buy SIGNAL (TD CLoP)"

        # SELL: Opens above yesterday’s open & close → then drops below both
        if (row["Open"] > prev_open and row["Open"] > prev_close
                and row["Low"] < prev_open and row["Low"] < prev_close):
            return "Sell SIGNAL (TD CLoP)"

        return ""

    intraday_df["TD CLoP"] = intraday_df.apply(check_td_clop, axis=1)
    return intraday_df


# ================================================================
#  DAY TYPE — At the 9:30 AM bar
# ================================================================
def td_day_type(intraday_df: pd.DataFrame,
                prev_high: float,
                prev_low: float) -> pd.DataFrame:
    """
    Labels the 9:30 bar as:
    - 'OUTSIDE (Above Prev High)'
    - 'OUTSIDE (Below Prev Low)'
    - 'WITHIN Range'
    """

    def determine_trap_status(open_price: float, p_high: float, p_low: float):

        if open_price is None or pd.isna(open_price):
            return ""
        if p_high is None or p_low is None:
            return "Unknown"
        if open_price > p_high:
            return "OUTSIDE (Above Prev High)"
        if open_price < p_low:
            return "OUTSIDE (Below Prev Low)"
        return "WITHIN Range"

    intraday_df["Day Type"] = ""

    mask_930 = intraday_df["Time"] == "09:30 AM"
    intraday_df.loc[mask_930, "Day Type"] = intraday_df[mask_930].apply(
        lambda row: determine_trap_status(row["Open"], prev_high, prev_low),
        axis=1,
    )

    return intraday_df


