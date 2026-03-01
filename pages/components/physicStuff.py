# # pages/components/physicStuff.py

# import pandas as pd
# import numpy as np


# def apply_physics_core(
#     df: pd.DataFrame,
#     rvol_col: str = "RVOL_5",
#     range_col: str = "Range",
#     vector_span: int = 3,
#     vol_window: int = 10,
#     gravity_threshold: float = 9.8,
# ) -> pd.DataFrame:
#     """
#     Core 'physics' layer for Mike.

#     Adds (if possible):

#       - Unit_pct              (numeric intrabar % move)
#       - Unit%                 (string, e.g. '+12%')
#       - Cumulative_Unit       (cumsum of Unit_pct)

#       - Vector_pct            (numeric 3-bar move, on last bar of each group)
#       - Vector%               (string 3-bar move, on last bar of each group)
#       - Vector_Momentum       (Vector_pct * 3-bar RVOL sum)
#       - Vector_Capacitance    (3-bar RVOL sum / |Vector_pct|)
#       - Vector_Efficiency     (Vector_Momentum / 3-bar price range sum)

#       - Volatility_Range      (rolling std of Range)
#       - Volatility_UnitVel    (rolling std of Unit velocity)
#       - Volatility_Composite  (Range + UnitVel vol)
#       - Volatility_Composite_Diff
#       - Gravity_Break_Alert   ('🪂' when composite jumps > gravity_threshold)
#     """
#     if df is None or df.empty:
#         # Return df with columns pre-created if you like, but empty is fine
#         return df

#     df = df.copy()

#     # -----------------------------------
#     # 1) Safety checks on required cols
#     # -----------------------------------
#     if not {"Open", "Close"}.issubset(df.columns):
#         # Can't do much without these
#         return df

#     if rvol_col not in df.columns:
#         # If no RVOL, create a dummy 1.0 column so formulas still work
#         df[rvol_col] = 1.0

#     # Ensure Range exists (High-Low) if missing
#     if range_col not in df.columns:
#         if {"High", "Low"}.issubset(df.columns):
#             df[range_col] = df["High"].astype(float) - df["Low"].astype(float)
#         else:
#             df[range_col] = 0.0

#     open_ = df["Open"].astype(float)
#     close = df["Close"].astype(float)
#     rvol = df[rvol_col].astype(float).fillna(0.0)
#     rng = df[range_col].astype(float).fillna(0.0)

#     # -----------------------------------
#     # 2) Unit% and Cumulative_Unit
#     # -----------------------------------
#     with np.errstate(divide="ignore", invalid="ignore"):
#         unit_pct = np.where(
#             open_ > 0,
#             ((close - open_) / open_) * 10000.0,
#             0.0,
#         )

#     unit_pct = np.nan_to_num(unit_pct, nan=0.0, posinf=0.0, neginf=0.0)

#     df["Unit_pct"] = unit_pct
#     df["Unit%"] = np.round(unit_pct).astype(int).astype(str) + "%"

#     df["Cumulative_Unit"] = df["Unit_pct"].cumsum()

#     # -----------------------------------
#     # 3) Vector% and vector metrics
#     #    (3-bar groups: 0-2, 3-5, 6-8, ...)
#     # -----------------------------------
#     n = len(df)
#     df["Vector_pct"] = np.nan
#     df["Vector%"] = ""
#     df["Vector_Momentum"] = np.nan
#     df["Vector_Capacitance"] = np.nan
#     df["Vector_Efficiency"] = np.nan

#     for i in range(vector_span - 1, n, vector_span):
#         i0 = i - (vector_span - 1)
#         i2 = i

#         if i0 < 0:
#             continue

#         o0 = open_.iloc[i0]
#         c2 = close.iloc[i2]

#         if not np.isfinite(o0) or not np.isfinite(c2) or o0 <= 0:
#             continue

#         # 3-bar displacement in "Mike" units
#         vec_pct = ((c2 - o0) / o0) * 10000.0

#         # 3-bar summed RVOL and Range
#         rvol_sum = rvol.iloc[i0:i2 + 1].sum()
#         range_sum = rng.iloc[i0:i2 + 1].sum()

#         # Store numeric + pretty string
#         df.iat[i2, df.columns.get_loc("Vector_pct")] = vec_pct
#         df.iat[i2, df.columns.get_loc("Vector%")] = f"{int(round(vec_pct, 0))}%"

#         # Momentum: power of the wave (displacement × fuel)
#         vec_mom = vec_pct * rvol_sum
#         df.iat[i2, df.columns.get_loc("Vector_Momentum")] = vec_mom

#         # Capacitance: volume per unit of displacement (coiled vs spent)
#         denom = abs(vec_pct)
#         cap = (rvol_sum / denom) if denom > 0 else np.nan
#         df.iat[i2, df.columns.get_loc("Vector_Capacitance")] = cap

#         # Efficiency: momentum per unit of 3-bar range
#         if range_sum and np.isfinite(range_sum) and range_sum != 0:
#             eff = vec_mom / range_sum
#         else:
#             eff = np.nan
#         df.iat[i2, df.columns.get_loc("Vector_Efficiency")] = eff

#     # -----------------------------------
#     # 4) Volatility composite + Gravity_Break_Alert
#     # -----------------------------------
#     # Unit "velocity" = change in Unit_pct bar-to-bar
#     unit_vel = pd.Series(df["Unit_pct"]).diff().fillna(0.0)

#     df["Volatility_Range"] = rng.rolling(vol_window, min_periods=1).std()
#     df["Volatility_UnitVel"] = unit_vel.rolling(vol_window, min_periods=1).std()

#     # Simple composite
#     df["Volatility_Composite"] = df["Volatility_Range"] + df["Volatility_UnitVel"]

#     df["Volatility_Composite_Diff"] = df["Volatility_Composite"].diff().fillna(0.0)

#     df["Gravity_Break_Alert"] = np.where(
#         df["Volatility_Composite_Diff"] > gravity_threshold,
#         "🪂",
#         "",
#     )

#     return df



import pandas as pd
import numpy as np

def apply_physics_core(
    df: pd.DataFrame,
    rvol_col: str = "RVOL_5",
    range_col: str = "Range",
    vector_span: int = 3,
    vol_window: int = 10,
    gravity_threshold: float = 9.8,
) -> pd.DataFrame:

    if df is None or df.empty:
        return df

    df = df.copy()

    # -------------------------------
    # 0) Required columns
    # -------------------------------
    if not {"Open", "Close"}.issubset(df.columns):
        return df

    # RVOL fallback
    if rvol_col not in df.columns:
        df[rvol_col] = 1.0

    # Range fallback
    if range_col not in df.columns:
        if {"High", "Low"}.issubset(df.columns):
            df[range_col] = df["High"].astype(float) - df["Low"].astype(float)
        else:
            df[range_col] = 0.0

    open_ = df["Open"].astype(float)
    close = df["Close"].astype(float)
    rvol = df[rvol_col].astype(float).fillna(0.0)
    rng = df[range_col].astype(float).fillna(0.0)

    # -------------------------------
    # 1) Unit_pct + Cumulative
    # -------------------------------
    with np.errstate(divide="ignore", invalid="ignore"):
        unit_pct = ((close - open_) / open_) * 10000.0
    unit_pct = np.nan_to_num(unit_pct, nan=0.0)

    df["Unit_pct"] = unit_pct
    df["Unit%"] = np.round(unit_pct).astype(int).astype(str) + "%"

    df["Cumulative_Unit"] = df["Unit_pct"].cumsum()

    # -------------------------------
    # 2) Prepare Vector columns
    # -------------------------------
    df["Vector_pct"]          = np.nan
    df["Vector%"]             = ""
    df["Vector_Momentum"]     = np.nan
    df["Vector_Capacitance"]  = np.nan
    df["Vector_Efficiency"]   = np.nan

    n = len(df)

    # -------------------------------
    # 3) Vector loop (3-bar waves)
    # -------------------------------
    for end_i in range(vector_span - 1, n, vector_span):
        start_i = end_i - (vector_span - 1)

        o0 = open_.iloc[start_i]
        c2 = close.iloc[end_i]

        if not (np.isfinite(o0) and np.isfinite(c2)) or o0 <= 0:
            continue

        vec_pct = ((c2 - o0) / o0) * 10000.0

        block_rvol  = rvol.iloc[start_i:end_i + 1].sum()
        block_range = rng.iloc[start_i:end_i + 1].sum()

        # Numeric + pretty
        df.at[end_i, "Vector_pct"] = vec_pct
        df.at[end_i, "Vector%"] = f"{int(round(vec_pct, 0))}%"

        # Momentum
        vec_mom = vec_pct * block_rvol
        df.at[end_i, "Vector_Momentum"] = vec_mom

        # Capacitance (volume per displacement)
        denom = abs(vec_pct)
        cap = (block_rvol / denom) if denom > 0 else np.nan
        df.at[end_i, "Vector_Capacitance"] = cap

        # Efficiency
        eff = vec_mom / block_range if block_range else np.nan
        df.at[end_i, "Vector_Efficiency"] = eff

    # -------------------------------
    # 4) Volatility & Gravity Alert
    # -------------------------------
    unit_vel = df["Unit_pct"].diff().fillna(0.0)

    df["Volatility_Range"]     = rng.rolling(vol_window, min_periods=1).std()
    df["Volatility_UnitVel"]   = unit_vel.rolling(vol_window, min_periods=1).std()

    comp = df["Volatility_Range"] + df["Volatility_UnitVel"]
    df["Volatility_Composite"] = comp

    df["Volatility_Composite_Diff"] = comp.diff().fillna(0.0)

    df["Gravity_Break_Alert"] = np.where(
        df["Volatility_Composite_Diff"] > gravity_threshold,
        "🪂",
        "",
    )

    return df
