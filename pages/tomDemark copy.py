import pandas as pd
import numpy as np

# ==========================================================
#  Tom Demark — GAP ANALYSIS
# ==========================================================
def td_gap_analysis(prev_close, prev_high, prev_low, intraday_df, gap_threshold_decimal):
    gap_alert = ""
    gap_type = None

    if prev_close is None or intraday_df.empty:
        return gap_alert, gap_type

    first_open = intraday_df["Open"].iloc[0]

    # fallback
    if pd.isna(first_open):
        first_open = prev_close

    gap_percentage = (first_open - prev_close) / prev_close

    # GAP UP
    if first_open > prev_high and gap_percentage > gap_threshold_decimal:
        gap_alert = "🚀 UP GAP ALERT"
        gap_type = "UP"

    # GAP DOWN
    elif first_open < prev_low and gap_percentage < -gap_threshold_decimal:
        gap_alert = "🔻 DOWN GAP ALERT"
        gap_type = "DOWN"

    return gap_alert, gap_type


# ==========================================================
#  Tom Demark — HIGH OF DAY / LOW OF DAY
# ==========================================================
def td_high_low_of_day(df):
    df = df.copy()

    # High of Day
    df["High of Day"] = ""
    for date_value, group_df in df.groupby("Date", as_index=False):
        indices = group_df.index
        current_high = -float("inf")
        last_high_row = None

        for idx in indices:
            this_high = df.loc[idx, "High"]
            if this_high > current_high:
                current_high = this_high
                last_high_row = idx
                df.at[idx, "High of Day"] = f"{current_high:.2f}"
            else:
                df.at[idx, "High of Day"] = f"+{idx - last_high_row}"

    # Low of Day
    df["Low of Day"] = ""
    for date_value, group_df in df.groupby("Date", as_index=False):
        indices = group_df.index
        current_low = float("inf")
        last_low_row = None

        for idx in indices:
            this_low = df.loc[idx, "Low"]
            if this_low < current_low:
                current_low = this_low
                last_low_row = idx
                df.at[idx, "Low of Day"] = f"{current_low:.2f}"
            else:
                df.at[idx, "Low of Day"] = f"+{idx - last_low_row}"

    return df
