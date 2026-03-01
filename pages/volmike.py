

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
import os
# ── 1. import at the top (native std-lib only)
import io, zipfile
import streamlit.components.v1 as components



# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.append(repo_root)
from pages.components.jsonExport import build_basic_json, render_json_batch_download
from pages.components.ichimokuLines import apply_ichimoku_f_levels
from pages.components.bollingerStuff import apply_bollinger_suite
from pages.components.stdExpansion import apply_std_expansion
from pages.components.rvolAlerts import apply_rvol_alerts
from pages.components.physicStuff import apply_physics_core
from pages.components.entrySystem import apply_entry_system
from pages.components.Door import apply_T0_door
from pages.components.T1Horse import apply_T1_horse
from pages.components.T2Lightning import apply_T2_lightning
from pages.components.parallel import apply_parallel_phase
from pages.components.goldmine import apply_goldmine_e1
from pages.components.e1EvilEye import apply_e1_kijun_evil_eye

from pages.components.midasAnchors import compute_midas_curves
from pages.components.marketProfile import compute_market_profile

from pages.components.gapSettings import get_gap_settings
# from pages.components.tomDemark import td_gap_analysis, td_high_low_of_day
from pages.components.demarkLowRiskEntries import (
    td_open_signals,
    td_trap_signals,
    td_clop_signals,
    td_clopwin_signals,
    td_day_type,
)
from pages.components.demarkSignals import (
    calculate_td_sequential,
    calculate_td_countdown,
    calculate_td_demand_supply_lines_fpercent,
    calculate_td_supply_cross_alert,
    calculate_clean_tdst,
    calculate_heaven_cloud,
    calculate_drizzle_emoji,
)

from pages.components.entrySystem import apply_entry_system

if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False


st.set_page_config(
    page_title="Volmike.com",
    layout="wide",
)

# =============================
# Sidebar Inputs
# =============================
st.sidebar.header("Input Options")

DEFAULT_TICKERS = [
  "ES=F",  "NQ=F","YM=F","SPY","VIXY","SOXX","NVDA","AMZN","MU","NQ=F","AMD","QCOM","SMCI","MSFT","uber", "AVGO","MRVL","QQQ","PLTR","AAPL","GOOGL","META","XLY","TSLA","nke","GM","c","DKNG","CHWY","ETSY","CART","W","KBE","wfc","hood","PYPL","coin","bac","jpm",
]

tickers = st.sidebar.multiselect(
    "Select Tickers",
    options=DEFAULT_TICKERS,
    default=["NVDA"],
)

start_date = st.sidebar.date_input("Start Date", value=date(2025, 11, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["2m", "5m", "15m", "30m", "60m", "1d"],
    index=1,
)

# =============================
# Helper Functions
# =============================
def calculate_f_numeric(df: pd.DataFrame, prev_close) -> pd.DataFrame:
    """Compute F_numeric as integer."""
    try:
        prev = float(prev_close)
    except Exception:
        df["F_numeric"] = 0
        return df

    if prev == 0 or df.empty or np.isnan(prev):
        df["F_numeric"] = 0
        return df

    f = ((df["Close"] - prev) / prev) * 10000
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["F_numeric"] = f.round(0).astype(int)
    return df


def calculate_f_percentage(df: pd.DataFrame, prev_close) -> pd.DataFrame:
    """Compute F% as text, for example '+44%'."""
    try:
        prev = float(prev_close)
    except Exception:
        df["F%"] = "N/A"
        return df

    if prev == 0 or df.empty or np.isnan(prev):
        df["F%"] = "N/A"
        return df

    f = ((df["Close"] - prev) / prev) * 10000
    f = f.replace([np.inf, -np.inf], np.nan).fillna(0)
    df["F%"] = f.round(0).astype(int).astype(str) + "%"
    return df


def fetch_prev_daily(ticker: str, ref_date: date):
    """Fetch previous daily bar before ref_date."""
    daily = yf.download(
        ticker,
        end=ref_date,
        interval="1d",
        progress=False,
    )

    if daily.empty:
        return {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "range": None,
        }

    if isinstance(daily.columns, pd.MultiIndex):
        daily.columns = [c[0] if isinstance(c, tuple) else c for c in daily.columns]

    prev_open = float(daily["Open"].iloc[-1])
    prev_high = float(daily["High"].iloc[-1])
    prev_low = float(daily["Low"].iloc[-1])
    prev_close = float(daily["Close"].iloc[-1])
    prev_range = prev_high - prev_low

    return {
        "open": prev_open,
        "high": prev_high,
        "low": prev_low,
        "close": prev_close,
        "range": prev_range,
    }


def fetch_intraday(ticker: str, start: date, end: date, tf: str) -> pd.DataFrame:
    """Fetch intraday data and normalize columns."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=tf,
        progress=False,
    )

    if df.empty:
        return df

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    # Timezone to US/Eastern, then make naive
    try:
        if df["Date"].dt.tz is None:
            df["Date"] = (
                df["Date"]
                .dt.tz_localize("UTC")
                .dt.tz_convert("America/New_York")
            )
        else:
            df["Date"] = df["Date"].dt.tz_convert("America/New_York")
    except Exception:
        # If anything weird happens, just leave as-is
        pass

    try:
        df["Date"] = df["Date"].dt.tz_localize(None)
    except Exception:
        pass

    df["Time"] = df["Date"].dt.strftime("%I:%M %p")
    df["Date"] = df["Date"].dt.date

    return df


def add_rvol(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add RVOL_5 to intraday data."""
    if df.empty or "Volume" not in df.columns:
        df["RVOL_5"] = np.nan
        return df

    if len(df) >= window:
        df["Avg_Vol_5"] = df["Volume"].rolling(window=window).mean()
        df["RVOL_5"] = df["Volume"] / df["Avg_Vol_5"]
        df.drop(columns=["Avg_Vol_5"], inplace=True)
    else:
        df["RVOL_5"] = np.nan

    return df


# def apply_td_gap_and_intraday_signals(
#     df: pd.DataFrame,
#     prev_open,
#     prev_high,
#     prev_low,
#     prev_close,
#     gap_threshold_decimal: float,
# ):
#     """Run Tom Demark gap and intraday signals and attach columns."""
#     gap_alert, gap_type = td_gap_analysis(
#         prev_close,
#         prev_high,
#         prev_low,
#         df,
#         gap_threshold_decimal,
#     )

#     df["TD Gap Alert"] = gap_alert
#     df["TD Gap Type"] = gap_type

#     # High / Low of Day marks
#     df = td_high_low_of_day(df)

#     # Entry / trap style signals
#     df = td_open_signals(df, prev_high, prev_low, gap_type)
#     df = td_trap_signals(df, prev_high, prev_low)
#     df = td_clop_signals(df, prev_open, prev_close)
#     df = td_day_type(df, prev_high, prev_low)
#     df = td_clopwin_signals(df)

#     return df, gap_alert, gap_type


def apply_td_advanced_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run TD Sequential, Countdown, Supply/Demand, TDST, Heaven and Drizzle."""
    df = calculate_td_sequential(df)
    df = calculate_td_countdown(df)
    df = calculate_td_demand_supply_lines_fpercent(df)
    df = calculate_td_supply_cross_alert(df)
    df = calculate_clean_tdst(df)
    df = calculate_heaven_cloud(df)
    df = calculate_drizzle_emoji(df)
    return df


# ============================================================
# Z3 MOMENTUM SCORE  (ported from mikeChart.js)
# ============================================================

def _robust_sigma_mad(arr: list) -> float | None:
    """MAD-based robust sigma estimator. Matches JS robustSigmaMAD."""
    a = [v for v in arr if v is not None and not np.isnan(v)]
    if len(a) < 5:
        return None
    med = float(np.median(a))
    mad = float(np.median([abs(v - med) for v in a]))
    if np.isnan(mad):
        return None
    return max(mad * 1.4826, 1e-6)   # MAD → sigma, never zero


def _bps_from_log_return(curr: float, prev: float) -> float | None:
    """Log return in basis points. Matches JS bpsFromLogReturn."""
    if curr is None or prev is None or curr <= 0 or prev <= 0:
        return None
    return np.log(curr / prev) * 10_000


def compute_z3_series(df: pd.DataFrame, sig_n: int = 9) -> pd.Series:
    """
    Rolling Z3 momentum score for every bar.
    Exactly mirrors JS z3Map logic:
      z3[i] = (b0 + b1 + b2) / (robustSigmaMAD(window) * √3)
    clamped to [-8, 8].
    Returns a float Series (NaN where not computable).
    """
    closes = pd.to_numeric(df["Close"], errors="coerce").values
    n = len(closes)
    SQRT3 = np.sqrt(3)

    # 1) BPS for each bar (log return × 10000)
    bps = [None] * n
    for i in range(1, n):
        bps[i] = _bps_from_log_return(closes[i], closes[i - 1])

    # 2) Z3 scores
    scores = [np.nan] * n
    for i in range(n):
        if i < sig_n + 2:
            continue
        window = [bps[j] for j in range(i - sig_n + 1, i + 1) if bps[j] is not None]
        if len(window) < max(5, int(sig_n * 0.6)):
            continue
        b0, b1, b2 = bps[i], bps[i - 1], bps[i - 2]
        if any(v is None for v in [b0, b1, b2]):
            continue
        sigma = _robust_sigma_mad(window)
        if sigma is None or sigma <= 0:
            continue
        z3 = (b0 + b1 + b2) / (sigma * SQRT3)
        scores[i] = max(-8.0, min(8.0, z3))

    return pd.Series(scores, index=df.index, name="Z3_Score")


def compute_mae_line(df: pd.DataFrame) -> dict | None:
    """
    Find MAE (Maximum Adverse Excursion) line from the latest blocked E1 (⏳).
    Mirrors JS maeLine useMemo.
    Returns dict with keys: e1_idx, mae_idx, end_idx, side, blocked_f, mae_f
    or None if no blocked E1 is found.
    """
    n = len(df)
    call_col = "Call_FirstEntry_Emoji" if "Call_FirstEntry_Emoji" in df.columns else None
    put_col  = "Put_FirstEntry_Emoji"  if "Put_FirstEntry_Emoji"  in df.columns else None
    if call_col is None and put_col is None:
        return None

    # 1) Scan backward for latest blocked E1
    e1_idx, side = -1, None
    for i in range(n - 1, -1, -1):
        if call_col and "⏳" in str(df[call_col].iloc[i]):
            e1_idx, side = i, "call"
            break
        if put_col and "⏳" in str(df[put_col].iloc[i]):
            e1_idx, side = i, "put"
            break

    if e1_idx < 0:
        return None

    # 2) Blocked entry F-level at the ⏳ bar
    blocked_f = float(df["F_numeric"].iloc[e1_idx])
    if np.isnan(blocked_f):
        return None

    # 3) End of window = first opposite *real* E1 (non-⏳, non-empty)
    end_idx = n - 1
    for j in range(e1_idx + 1, n):
        if side == "put" and call_col:
            val = str(df[call_col].iloc[j])
            if val and val not in ("", "nan") and "⏳" not in val:
                end_idx = j
                break
        if side == "call" and put_col:
            val = str(df[put_col].iloc[j])
            if val and val not in ("", "nan") and "⏳" not in val:
                end_idx = j
                break

    # 4) MAE = worst F_numeric from e1_idx → end_idx
    mae_f, mae_idx = blocked_f, e1_idx
    for i in range(e1_idx, end_idx + 1):
        f = float(df["F_numeric"].iloc[i])
        if np.isnan(f):
            continue
        if side == "call" and f < mae_f:
            mae_f, mae_idx = f, i
        elif side == "put" and f > mae_f:
            mae_f, mae_idx = f, i

    return {
        "e1_idx":   e1_idx,
        "mae_idx":  mae_idx,
        "end_idx":  end_idx,
        "side":     side,
        "blocked_f": blocked_f,
        "mae_f":    mae_f,
    }


def apply_z3_key(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Compute Z3 scores and mark the Z3 Key (🔑):
    the FIRST bar after MAE where Z3 crosses into the favorable ignition zone
    (z >= +threshold for call, z <= -threshold for put).
    Adds columns: 'Z3_Score', 'Z3_Key_Emoji', 'Z3_Key_Side'.
    Mirrors JS z3Key useMemo.
    """
    df = df.copy()
    df["Z3_Score"]    = compute_z3_series(df)
    df["Z3_Key_Emoji"] = ""
    df["Z3_Key_Side"]  = ""

    mae = compute_mae_line(df)
    if mae is None:
        return df

    side     = mae["side"]
    mae_idx  = mae["mae_idx"]
    end_idx  = mae["end_idx"]
    scores   = df["Z3_Score"].values

    def ignited(z: float) -> bool:
        if np.isnan(z):
            return False
        return z >= threshold if side == "call" else z <= -threshold

    # Scan from first bar AFTER MAE tick
    start = mae_idx + 1
    if start > end_idx:
        return df

    for i in range(start, end_idx + 1):
        z0 = scores[i]
        z1 = scores[i - 1] if i > 0 else np.nan
        if not ignited(z1) and ignited(z0):
            df.iat[i, df.columns.get_loc("Z3_Key_Emoji")] = "🔑"
            df.iat[i, df.columns.get_loc("Z3_Key_Side")]  = side
            break  # only the FIRST ignition crossing

    return df


def compute_initial_balance(df: pd.DataFrame, bars: int = 12):
    """Compute IB High/Low and thirds using first `bars` bars of F_numeric."""
    if df.empty or "F_numeric" not in df.columns or len(df) < bars:
        return {
            "ib_high": None,
            "ib_low": None,
            "ib_mid_third": None,
            "ib_upper_third": None,
        }

    ib_data = df.iloc[:bars]
    ib_high = ib_data["F_numeric"].max()
    ib_low = ib_data["F_numeric"].min()

    width = ib_high - ib_low
    third = width / 3 if width != 0 else 0
    ib_mid = ib_low + third
    ib_top = ib_low + 2 * third

    return {
        "ib_high": ib_high,
        "ib_low": ib_low,
        "ib_mid_third": ib_mid,
        "ib_upper_third": ib_top,
    }


def build_letter_profile(df: pd.DataFrame, mike_col: str = "F_numeric") -> pd.DataFrame:
    """Build a simple TPO-style profile using letters and tail flags."""
    if df.empty or mike_col not in df.columns:
        return pd.DataFrame(columns=["F% Level", "Letters", "Tail", "FirstTime"])

    # 1) F% bins
    f_bins = np.arange(-400, 401, 20)
    df = df.copy()
    df["F_Bin"] = pd.cut(
        df[mike_col],
        bins=f_bins,
        labels=[str(x) for x in f_bins[:-1]],
    )

    # 2) TimeIndex for 15-min letters
    df["TimeIndex"] = pd.to_datetime(
        df["Time"],
        format="%I:%M %p",
        errors="coerce",
    )
    df = df[df["TimeIndex"].notna()]

    df["LetterIndex"] = (
        (df["TimeIndex"].dt.hour * 60 + df["TimeIndex"].dt.minute) // 15
    ).astype(int)
    df["LetterIndex"] -= df["LetterIndex"].min()

    import string

    letters = string.ascii_uppercase

    def letter_code(n: int) -> str:
        n = int(n)
        if n < 26:
            return letters[n]
        first = letters[(n // 26) - 1]
        second = letters[n % 26]
        return first + second

    df["Letter"] = df["LetterIndex"].apply(letter_code)

    profile = {}
    for f_bin in f_bins[:-1]:
        f_bin_str = str(f_bin)
        lvl_letters = (
            df.loc[df["F_Bin"] == f_bin_str, "Letter"]
            .dropna()
            .unique()
        )
        if len(lvl_letters) > 0:
            profile[f_bin_str] = "".join(sorted(lvl_letters))

    profile_df = pd.DataFrame(profile.items(), columns=["F% Level", "Letters"])
    if profile_df.empty:
        profile_df["Tail"] = []
        profile_df["FirstTime"] = []
        return profile_df

    # First time at each F% level
    first_times = []
    for _, row in profile_df.iterrows():
        lvl_str = row["F% Level"]
        mask = df["F_Bin"] == str(lvl_str)
        lvl_rows = df[mask].sort_values("TimeIndex")
        if lvl_rows.empty:
            first_times.append(None)
        else:
            first_times.append(lvl_rows["Time"].iloc[0])

    profile_df["FirstTime"] = first_times
    profile_df["F% Level"] = profile_df["F% Level"].astype(int)
    profile_df["Tail"] = profile_df["Letters"].apply(
        lambda x: "🪶" if isinstance(x, str) and len(set(x)) == 1 else ""
    )

    return profile_df


def build_chart(
    intraday: pd.DataFrame,
    ib_stats: dict,
    profile_df_letters: pd.DataFrame,
    mp_df: pd.DataFrame,
) -> go.Figure:
    """Build main F_numeric chart with TD lines, MIDAS, IB, tails, Nose and Ear."""
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
    go.Scatter(
        x=intraday["Time"],
        y=intraday["F_numeric"],
        mode="lines+markers",
        line=dict(width=2, color="#48befe"),
        name="F_numeric",
        customdata=np.stack(
            [
                intraday["Close"],      # [0]
                intraday["Volume"],     # [1]
                intraday["RVOL_5"],     # [2]
            ],
            axis=-1,
        ),
        hovertemplate=(
            "Time: %{x}<br>"
            "F: %{y:.0f}<br>"
            "Close: %{customdata[0]:.2f}<br>"
            "Volume: %{customdata[1]:,.0f}<br>"
            "RVOL_5: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
    ),
    row=1, col=1,
)

    # ==========================
    # ♕ QUEEN (Kijun Cross)
    # ==========================
    if "Kijun_F" in intraday.columns and "F_numeric" in intraday.columns:
        mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
        kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")

        for i in range(1, len(intraday)):
            m0, m1 = mike.iloc[i], mike.iloc[i - 1]
            k0, k1 = kijun.iloc[i], kijun.iloc[i - 1]
            if any(pd.isna(v) for v in [m0, m1, k0, k1]):
                continue

            up_cross   = m1 < k1 and m0 >= k0
            down_cross = m1 > k1 and m0 <= k0
            if not up_cross and not down_cross:
                continue

            color = "#22c55e" if up_cross else "#ff3b3b"

            fig.add_trace(
                go.Scatter(
                    x=[intraday["Time"].iloc[i]],
                    y=[mike.iloc[i]],
                    mode="text",
                    text=["♕"],
                    textposition="middle center",
                    textfont=dict(size=30, color=color),
                    name="♕",
                    showlegend=False,
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "F%: %{y}<br>"
                        f"♕ {'↑' if up_cross else '↓'}"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )
    # TD Supply line (F%)
    if "TD Supply Line F" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["TD Supply Line F"],
                mode="lines",
                line=dict(width=0.3, color="#8A2BE2", dash="dot"),
                name="TD Supply F%",
                hovertemplate="Time: %{x}<br>Supply (F%): %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # TD Demand line (F%)
    if "TD Demand Line F" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["TD Demand Line F"],
                mode="lines",
                line=dict(width=0.3, color="#5DADE2", dash="dot"),
                name="TD Demand F%",
                hovertemplate="Time: %{x}<br>Demand (F%): %{y:.2f}<extra></extra>",
            ),
            row=1, col=1,
        )



    # MIDAS curves
    if "MIDAS_Bear" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["MIDAS_Bear"],
                mode="lines",
                line=dict(color="#ff4d4d", width=1.5, dash="longdash"),
                name="MIDAS Bear",
            ),
            row=1, col=1,
        )

    if "MIDAS_Bull" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["MIDAS_Bull"],
                mode="lines",
                line=dict(color="#2ecc71", width=1.5, dash="longdash"),
                name="MIDAS Bull",
            ),
            row=1, col=1,
        )


 # Ichimoku F-lines (Kijun & Tenkan)
    if "Kijun_F" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["Kijun_F"],
                mode="lines",
                line=dict(color="#2ecc71", width=0.7),  # orange-ish
                name="Kijun_F",
            ),
            row=1, col=1,
        )

    if "F% Tenkan" in intraday.columns:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=intraday["F% Tenkan"],
                mode="lines",
                line=dict(color="#ff4d4d", width=0.7),  # blue-ish
                name="F% Tenkan",
            ),
            row=1, col=1,
        )



    if {"F% Upper", "F% Lower", "F% MA"}.issubset(intraday.columns):
        # (B) Upper Band
        upper_band = go.Scatter(
            x=intraday["Time"],
            y=intraday["F% Upper"],
            mode="lines",
            line=dict(dash="solid", color="#d3d3d3", width=1),
            name="Upper Band",
        )

        # (C) Lower Band
        lower_band = go.Scatter(
            x=intraday["Time"],
            y=intraday["F% Lower"],
            mode="lines",
            line=dict(dash="solid", color="#d3d3d3", width=1),
            name="Lower Band",
        )

        # (D) Moving Average (Middle Band)
        middle_band = go.Scatter(
            x=intraday["Time"],
            y=intraday["F% MA"],
            mode="lines",
            line=dict(dash="dash", color="#d3d3d3", width=2),
            name="Middle Band (20-MA)",  # <- matches your window=20
        )

        fig.add_trace(upper_band, row=1, col=1)
        fig.add_trace(lower_band, row=1, col=1)
        fig.add_trace(middle_band, row=1, col=1)



    # 🐝 BBW Tight → ♗ above/below Mike
    if "BBW_Tight_Emoji" in intraday.columns:
        mask_bbw_tight = intraday["BBW_Tight_Emoji"] == "🐝"

        if mask_bbw_tight.any():
            if "Kijun_F" in intraday.columns:
                mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                above_mask = mask_bbw_tight & (mike >= kijun)
                below_mask = mask_bbw_tight & (mike <  kijun)
            else:
                above_mask = mask_bbw_tight
                below_mask = pd.Series(False, index=intraday.index)

            offset = 50

            for mask, pos_offset, textpos in [
                (above_mask, +offset, "top center"),
                (below_mask, -offset, "bottom center"),
            ]:
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[mask, "Time"],
                            y=intraday.loc[mask, "F_numeric"] + pos_offset,
                            mode="text",
                            text=["♗"] * int(mask.sum()),
                            textposition=textpos,
                            textfont=dict(size=22, color="#FACC15"),
                            name="BBW Tight ♗",
                            showlegend=False,
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y:.2f}<br>"
                                "BBW Tight Squeeze ♗<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )


        # BBW Expansion ♗
    if "BBW Alert" in intraday.columns:
        mask_bbw_alert = intraday["BBW Alert"] != ""

        if mask_bbw_alert.any():
            if "Kijun_F" in intraday.columns:
                mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                above_mask = mask_bbw_alert & (mike >= kijun)
                below_mask = mask_bbw_alert & (mike <  kijun)
            else:
                above_mask = mask_bbw_alert
                below_mask = pd.Series(False, index=intraday.index)

            for mask, pos_offset, textpos, color in [
                (above_mask, +50, "top center",    "#22c55e"),
                (below_mask, -50, "bottom center", "#ff3b3b"),
            ]:
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[mask, "Time"],
                            y=intraday.loc[mask, "F_numeric"] + pos_offset,
                            mode="text",
                            text=["♗"] * int(mask.sum()),
                            textposition=textpos,
                            textfont=dict(size=24, color=color),
                            name="BBW Expansion ♗",
                            showlegend=False,
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "♗ BBW Expansion<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )
    # ==========================
    # Marengo ♞ (North & South)
    # ==========================
    if "Marengo" in intraday.columns:
        marengo_mask = intraday["Marengo"] == "🐎"
        if marengo_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=intraday.loc[marengo_mask, "Time"],
                    y=intraday.loc[marengo_mask, "F_numeric"] + 100,
                    mode="text",
                    text=["♞"] * int(marengo_mask.sum()),
                    textfont=dict(size=24, color="#22D3EE"),
                    textposition="top center",
                    name="Marengo ♞",
                    showlegend=False,
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>♞ Marengo Up<extra></extra>",
                ),
                row=1, col=1,
            )

    if "South_Marengo" in intraday.columns:
        south_mask = intraday["South_Marengo"] == "🐎"
        if south_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=intraday.loc[south_mask, "Time"],
                    y=intraday.loc[south_mask, "F_numeric"] - 100,
                    mode="text",
                    text=["♞"] * int(south_mask.sum()),
                    textfont=dict(size=24, color="#38BDF8"),
                    textposition="bottom center",
                    name="South Marengo ♞",
                    showlegend=False,
                    hovertemplate="Time: %{x}<br>F%: %{y}<br>♞ Marengo Down<extra></extra>",
                ),
                row=1, col=1,
            )

    # 🟢 STD Expansion (♗)
    if "STD_Alert" in intraday.columns:
        mask_std_alert = intraday["STD_Alert"] != ""

        if mask_std_alert.any():
            if "Kijun_F" in intraday.columns:
                mike   = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun  = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                above_mask = mask_std_alert & (mike >= kijun)
                below_mask = mask_std_alert & (mike <  kijun)
            else:
                above_mask = mask_std_alert
                below_mask = pd.Series(False, index=intraday.index)

            offset = 32

            for mask, pos_offset, textpos in [
                (above_mask, +offset, "top center"),
                (below_mask, -offset, "bottom center"),
            ]:
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[mask, "Time"],
                            y=intraday.loc[mask, "F_numeric"] + pos_offset,
                            mode="text",
                            text=["♗"] * int(mask.sum()),
                            textposition=textpos,
                            textfont=dict(size=24, color="#BF40BF"),
                            name="STD Expansion ♗",
                            showlegend=False,
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "STD Expansion ♗<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )


    # Initial Balance lines (manual F_numeric)
    ib_high = ib_stats.get("ib_high")
    ib_low = ib_stats.get("ib_low")
    ib_mid_third = ib_stats.get("ib_mid_third")
    ib_upper_third = ib_stats.get("ib_upper_third")

    if ib_high is not None and ib_low is not None:
        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=[ib_high] * len(intraday),
                mode="lines",
                line=dict(color="#FFD700", dash="dot", width=0.7),
                name="IB High",
                showlegend=True,
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=intraday["Time"],
                y=[ib_low] * len(intraday),
                mode="lines",
                line=dict(color="#FFD700", dash="dot", width=0.7),
                name="IB Low",
                showlegend=True,
            ),
            row=1, col=1,
        )

        if ib_mid_third is not None:
            fig.add_trace(
                go.Scatter(
                    x=intraday["Time"],
                    y=[ib_mid_third] * len(intraday),
                    mode="lines",
                    line=dict(color="#d3d3d3", dash="dot", width=0.2),
                    name="IB Mid Third",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        if ib_upper_third is not None:
            fig.add_trace(
                go.Scatter(
                    x=intraday["Time"],
                    y=[ib_upper_third] * len(intraday),
                    mode="lines",
                    line=dict(color="#d3d3d3", dash="dot", width=0.2),
                    name="IB Upper Third",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # 🪶 Tails from simple letter profile
    if not profile_df_letters.empty:
        for _, row in profile_df_letters.iterrows():
            if row.get("Tail") == "🪶":
                f_level = row["F% Level"]
                first_time = row.get("FirstTime")
                if first_time is None:
                    time_at_level = intraday["Time"].iloc[-1]
                else:
                    time_at_level = first_time

                fig.add_trace(
                    go.Scatter(
                        x=[time_at_level],
                        y=[f_level],
                        mode="text",
                        text=["🪶"],
                        textposition="middle right",
                        textfont=dict(size=16),
                        name="🪶 Tail",
                        showlegend=False,
                        hovertemplate=(
                            "🪶 Tail<br>"
                            f"F% Level: {f_level}<br>"
                            f"Time: {time_at_level}<extra></extra>"
                        ),
                    ),
                    row=1, col=1,
                )

    # 👃🏽 Nose and 🦻🏼 Ear from market profile df
    if not mp_df.empty:
        # Ensure columns exist
        for col in ["F% Level", "TPO_Count", "%Vol", "🦻🏼", "👃🏽", "Time"]:
            if col not in mp_df.columns:
                mp_df[col] = 0 if col in ["TPO_Count", "%Vol"] else ""

        # Nose POC (by time / letters)
        nose_row = mp_df.sort_values(by="TPO_Count", ascending=False).head(1)
        if not nose_row.empty:
            poc_f_level = int(nose_row["F% Level"].iat[0])
            nose_time = nose_row["Time"].iat[0]

            fig.add_hline(
                y=poc_f_level,
                line_color="#ff1493",
                line_dash="dot",
                line_width=0.2,
                row=1,
                col=1,
                # annotation_text="👃🏽 Nose POC",
                annotation_position="top right",
                annotation_font_color="#ff1493",
                showlegend=False,
            )





    # # =============================
    # # 🎯 ENTRY MARKERS (PUT / CALL)
    # # =============================

    # # --- PUT 🎯 / 🎯2 / 🎯3 ---
    # if "Put_FirstEntry_Emoji" in intraday.columns:
    #     first_entry_mask = intraday["Put_FirstEntry_Emoji"] == "🎯"
    #     if first_entry_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[first_entry_mask, "Time"],
    #                 y=intraday.loc[first_entry_mask, "F_numeric"] - 34,
    #                 mode="text",
    #                 text=intraday.loc[first_entry_mask, "Put_FirstEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯 Put Entry 1",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )

    # if "Put_SecondEntry_Emoji" in intraday.columns:
    #     second_entry_mask = intraday["Put_SecondEntry_Emoji"] == "🎯2"
    #     if second_entry_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[second_entry_mask, "Time"],
    #                 y=intraday.loc[second_entry_mask, "F_numeric"] - 34,
    #                 mode="text",
    #                 text=intraday.loc[second_entry_mask, "Put_SecondEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯2 Put Entry 2",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )

    # if "Put_ThirdEntry_Emoji" in intraday.columns:
    #     third_entry_mask = intraday["Put_ThirdEntry_Emoji"] == "🎯3"
    #     if third_entry_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[third_entry_mask, "Time"],
    #                 y=intraday.loc[third_entry_mask, "F_numeric"] - 34,
    #                 mode="text",
    #                 text=intraday.loc[third_entry_mask, "Put_ThirdEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯3 Put Entry 3",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )

    # # --- CALL 🎯 / 🎯2 / 🎯3 ---
    # if "Call_FirstEntry_Emoji" in intraday.columns:
    #     call1_mask = intraday["Call_FirstEntry_Emoji"] == "🎯"
    #     if call1_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[call1_mask, "Time"],
    #                 y=intraday.loc[call1_mask, "F_numeric"] + 34,
    #                 mode="text",
    #                 text=intraday.loc[call1_mask, "Call_FirstEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯 Call Entry 1",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )

    # if "Call_SecondEntry_Emoji" in intraday.columns:
    #     call2_mask = intraday["Call_SecondEntry_Emoji"] == "🎯2"
    #     if call2_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[call2_mask, "Time"],
    #                 y=intraday.loc[call2_mask, "F_numeric"] + 34,
    #                 mode="text",
    #                 text=intraday.loc[call2_mask, "Call_SecondEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯2 Call Entry 2",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )

    # if "Call_ThirdEntry_Emoji" in intraday.columns:
    #     call3_mask = intraday["Call_ThirdEntry_Emoji"] == "🎯3"
    #     if call3_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[call3_mask, "Time"],
    #                 y=intraday.loc[call3_mask, "F_numeric"] + 34,
    #                 mode="text",
    #                 text=intraday.loc[call3_mask, "Call_ThirdEntry_Emoji"],
    #                 textposition="top center",
    #                 textfont=dict(size=24),
    #                 name="🎯3 Call Entry 3",
    #                 showlegend=False,
    #                 hovertemplate="Time: %{x}<br>F%: %{y}<extra></extra>",
    #             ),
    #             row=1, col=1,
    #         )



    # =============================
    # 🎯 ENTRY MARKERS (PUT / CALL)
    # =============================

    for side, offset, cols in [
        ("call", +34, {
            "exec":    "Call_FirstEntry_Emoji",
            "hold":    "Call_FirstEntry_Emoji",
            "reclaim": "Call_DeferredEntry_Emoji",
            "e2":      "Call_SecondEntry_Emoji",
            "e3":      "Call_ThirdEntry_Emoji",
        }),
        ("put", -34, {
            "exec":    "Put_FirstEntry_Emoji",
            "hold":    "Put_FirstEntry_Emoji",
            "reclaim": "Put_DeferredEntry_Emoji",
            "e2":      "Put_SecondEntry_Emoji",
            "e3":      "Put_ThirdEntry_Emoji",
        }),
    ]:
        label = side.upper()

        # 🎯 EXEC
        col = cols["exec"]
        if col in intraday.columns:
            mask = intraday[col] == "🎯"
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[mask, "Time"],
                        y=intraday.loc[mask, "F_numeric"] + offset,
                        mode="text",
                        text=["🎯"] * int(mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=24),
                        name=f"🎯 E1 {label}",
                        showlegend=False,
                        hovertemplate=f"Time: %{{x}}<br>F%: %{{y}}<br>🎯 Entry 1 {label}<extra></extra>",
                    ),
                    row=1, col=1,
                )

        # ⏳ HOLD
        col = cols["hold"]
        if col in intraday.columns:
            mask = intraday[col] == "⏳"
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[mask, "Time"],
                        y=intraday.loc[mask, "F_numeric"] + offset,
                        mode="text",
                        text=["⏳"] * int(mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=24),
                        name=f"⏳ HOLD {label}",
                        showlegend=False,
                        hovertemplate=f"Time: %{{x}}<br>F%: %{{y}}<br>⏳ Blocked E1 {label}<extra></extra>",
                    ),
                    row=1, col=1,
                )

        # 🧿 RECLAIM
        col = cols["reclaim"]
        if col in intraday.columns:
            mask = intraday[col] == "🧿"
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[mask, "Time"],
                        y=intraday.loc[mask, "F_numeric"] + offset,
                        mode="text",
                        text=["🧿"] * int(mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=24),
                        name=f"🧿 RECLAIM {label}",
                        showlegend=False,
                        hovertemplate=f"Time: %{{x}}<br>F%: %{{y}}<br>🧿 Reclaim {label}<extra></extra>",
                    ),
                    row=1, col=1,
                )

        # 🎯2 E2
        col = cols["e2"]
        if col in intraday.columns:
            mask = intraday[col] == "🎯2"
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[mask, "Time"],
                        y=intraday.loc[mask, "F_numeric"] + offset,
                        mode="text",
                        text=["🎯2"] * int(mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=24),
                        name=f"🎯2 E2 {label}",
                        showlegend=False,
                        hovertemplate=f"Time: %{{x}}<br>F%: %{{y}}<br>🎯2 Entry 2 {label}<extra></extra>",
                    ),
                    row=1, col=1,
                )

        # 🎯3 E3
        col = cols["e3"]
        if col in intraday.columns:
            mask = intraday[col] == "🎯3"
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[mask, "Time"],
                        y=intraday.loc[mask, "F_numeric"] + offset,
                        mode="text",
                        text=["🎯3"] * int(mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=24),
                        name=f"🎯3 E3 {label}",
                        showlegend=False,
                        hovertemplate=f"Time: %{{x}}<br>F%: %{{y}}<br>🎯3 Entry 3 {label}<extra></extra>",
                    ),
                    row=1, col=1,
                )




    # ==========================
    # 🏇🏼 T1 HORSE MARKER
    # ==========================
    if "T1_Emoji" in intraday.columns:
        t1_mask = intraday["T1_Emoji"] == "🏇🏼"

        if t1_mask.any():
            offset = 50  # adjust height above/below Mike

            if "Kijun_F" in intraday.columns:
                mike = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"], errors="coerce")

                # T1 positions
                below_mask = t1_mask & (mike < kijun)    # Mike under Kijun → horse under Mike
                above_mask = t1_mask & (mike >= kijun)   # Mike above Kijun → horse above Mike

                # 🏇🏼 UNDER Mike
                if below_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[below_mask, "Time"],
                            y=intraday.loc[below_mask, "F_numeric"] - offset,
                            mode="text",
                            text=["🏇🏼"] * int(below_mask.sum()),
                            textposition="middle center",
                            textfont=dict(size=22, color="mediumvioletred"),
                            name="T1 Horse",
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "T1 Horse 🏇🏼<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                # ==========================
    # 🏇🏼 T1 HORSE MARKER
    # ==========================
    if "T1_Emoji" in intraday.columns:
        t1_mask = intraday["T1_Emoji"] == "🏇🏼"

        if t1_mask.any():
            offset = 120  # ← was 50, pushed further out

            if "Kijun_F" in intraday.columns:
                mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                below_mask = t1_mask & (mike <  kijun)
                above_mask = t1_mask & (mike >= kijun)

                if below_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[below_mask, "Time"],
                            y=intraday.loc[below_mask, "F_numeric"] - offset,
                            mode="text",
                            text=["🏇🏼"] * int(below_mask.sum()),
                            textposition="middle center",
                            textfont=dict(size=36, color="mediumvioletred"),  # ← was 22
                            name="T1 Horse",
                            showlegend=False,
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>T1 Horse 🏇🏼<extra></extra>",
                        ),
                        row=1, col=1,
                    )

                if above_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[above_mask, "Time"],
                            y=intraday.loc[above_mask, "F_numeric"] + offset,
                            mode="text",
                            text=["🏇🏼"] * int(above_mask.sum()),
                            textposition="middle center",
                            textfont=dict(size=36, color="mediumvioletred"),  # ← was 29
                            name="T1 Horse",
                            showlegend=False,
                            hovertemplate="Time: %{x}<br>F%: %{y}<br>T1 Horse 🏇🏼<extra></extra>",
                        ),
                        row=1, col=1,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[t1_mask, "Time"],
                        y=intraday.loc[t1_mask, "F_numeric"] + offset,
                        mode="text",
                        text=["🏇🏼"] * int(t1_mask.sum()),
                        textposition="middle center",
                        textfont=dict(size=36, color="mediumvioletred"),
                        name="T1 Horse",
                        showlegend=False,
                        hovertemplate="Time: %{x}<br>F%: %{y}<br>T1 Horse 🏇🏼<extra></extra>",
                    ),
                    row=1, col=1,
                )


    # ==========================
    # ⚡ T2 LIGHTNING MARKER
    # ==========================
    if "T2_Emoji" in intraday.columns:
        t2_mask = intraday["T2_Emoji"] == "⚡"

        if t2_mask.any():
            offset = 14  # height spacing above/below Mike

            if "Kijun_F" in intraday.columns:
                mike = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"], errors="coerce")

                below_mask = t2_mask & (mike < kijun)
                above_mask = t2_mask & (mike >= kijun)

                # ⚡ BELOW Mike (Mike < Kijun)
                if below_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[below_mask, "Time"],
                            y=intraday.loc[below_mask, "F_numeric"] - offset,
                            mode="text",
                            text=["⚡"] * int(below_mask.sum()),
                            textposition="middle left",
                            textfont=dict(size=16, color="gold"),
                            name="T2 Lightning",
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "T2 Lightning ⚡<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                # ⚡ ABOVE Mike (Mike ≥ Kijun)
                if above_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[above_mask, "Time"],
                            y=intraday.loc[above_mask, "F_numeric"] + offset,
                            mode="text",
                            text=["⚡"] * int(above_mask.sum()),
                            textposition="middle left",
                            textfont=dict(size=16, color="gold"),
                            name="T2 Lightning",
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "T2 Lightning ⚡<extra></extra>"
                            ),
                        ),
                        row=1,
                        col=1,
                    )

            else:
                # fallback: above Mike if Kijun missing
                fig.add_trace(
                    go.Scatter(
                        x=intraday.loc[t2_mask, "Time"],
                        y=intraday.loc[t2_mask, "F_numeric"] + offset,
                        mode="text",
                        text=["⚡"] * int(t2_mask.sum()),
                        textposition="middle left",
                        textfont=dict(size=16, color="gold"),
                        name="T2 Lightning",
                        hovertemplate=(
                            "Time: %{x}<br>"
                            "F%: %{y}<br>"
                            "T2 Lightning ⚡<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

# ==========================
    # ♚ KING (T2 bars)
    # ==========================
    if "T2_Emoji" in intraday.columns:
        king_mask = intraday["T2_Emoji"] == "⚡"
        if king_mask.any():
            if "Kijun_F" in intraday.columns:
                mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                above_mask = king_mask & (mike >= kijun)
                below_mask = king_mask & (mike <  kijun)
            else:
                above_mask = king_mask
                below_mask = pd.Series(False, index=intraday.index)

            for mask, pos_offset, textpos in [
                (above_mask, +94, "top center"),
                (below_mask, -94, "bottom center"),
            ]:
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[mask, "Time"],
                            y=intraday.loc[mask, "F_numeric"] + pos_offset,
                            mode="text",
                            text=["♚"] * int(mask.sum()),
                            textposition=textpos,
                            textfont=dict(size=44, color="#F59E0B"),
                            name="King ♚",
                            showlegend=False,
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "♚ KING<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )




    # ==========================
    # MIKE LINE (Z3 engine paint)
    # ==========================
    TH = 1.5
    MIKE_BLUE  = "#1E90FF"
    MIKE_GREEN = "#22c55e"
    MIKE_RED   = "#ff3b3b"

    def color_for_z3(z):
        if pd.notna(z):
            if z >= TH:  return MIKE_GREEN
            if z <= -TH: return MIKE_RED
        return MIKE_BLUE

    # Build color per bar
    z3_scores = intraday["Z3_Score"] if "Z3_Score" in intraday.columns else pd.Series(float("nan"), index=intraday.index)
    bar_colors = [color_for_z3(z) for z in z3_scores]

    # Split into contiguous same-color segments
    segments = []
    seg_start = 0
    for i in range(1, len(intraday)):
        if bar_colors[i] != bar_colors[i - 1]:
            segments.append((seg_start, i, bar_colors[i - 1]))
            seg_start = i - 1  # overlap by 1 so segments connect
    segments.append((seg_start, len(intraday) - 1, bar_colors[-1]))

    for seg_start, seg_end, color in segments:
        seg = intraday.iloc[seg_start: seg_end + 1]
        if len(seg) < 2:
            continue
        fig.add_trace(
            go.Scatter(
                x=seg["Time"],
                y=seg["F_numeric"],
                mode="lines",
                line=dict(width=2, color=color),
                showlegend=False,
                name="Mike",
                hovertemplate=(
                    "Time: %{x}<br>"
                    "F: %{y:.0f}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Dots on top (separate trace, always blue so they don't override segment color)
    fig.add_trace(
        go.Scatter(
            x=intraday["Time"],
            y=intraday["F_numeric"],
            mode="markers",
            marker=dict(size=3, color=bar_colors),
            showlegend=False,
            name="Mike dots",
            customdata=np.stack([
                intraday["Close"],
                intraday["Volume"],
                intraday["RVOL_5"],
            ], axis=-1),
            hovertemplate=(
                "Time: %{x}<br>"
                "F: %{y:.0f}<br>"
                "Close: %{customdata[0]:.2f}<br>"
                "Volume: %{customdata[1]:,.0f}<br>"
                "RVOL_5: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )


    # ==========================
    # 🏁 / 🚩 PARALLEL PHASE (start + end flags)
    # ==========================
    if "Parallel_Emoji" in intraday.columns:
        par_mask = intraday["Parallel_Emoji"] == "⚡"
        if par_mask.any():
            par_bars = intraday[par_mask]
            first_par = par_bars.iloc[0]

            # 🏁 Start — first bar inside the phase
            fig.add_trace(
                go.Scatter(
                    x=[first_par["Time"]],
                    y=[first_par["F_numeric"] + 20],
                    mode="text",
                    text=["🏁"],
                    textposition="middle center",
                    textfont=dict(size=20),
                    name="Parallel Start 🏁",
                    hovertemplate="🏁 Parallel Start<br>Time: %{x}<br>F%: %{y}<extra></extra>",
                ),
                row=1, col=1,
            )

            # 🚩 End — first bar AFTER the phase breaks (Mike crossed Tenkan)
            last_par_loc = intraday.index.get_loc(par_bars.index[-1])
            next_loc = last_par_loc + 1

            if next_loc < len(intraday):
                break_bar = intraday.iloc[next_loc]
                fig.add_trace(
                    go.Scatter(
                        x=[break_bar["Time"]],
                        y=[break_bar["F_numeric"] + 20],
                        mode="text",
                        text=["🚩"],
                        textposition="middle center",
                        textfont=dict(size=20),
                        name="Parallel End 🚩",
                        hovertemplate="🚩 Parallel Break<br>Time: %{x}<br>F%: %{y}<extra></extra>",
                    ),
                    row=1, col=1,
                )
    # # ==========================
    # # 💰 Goldmine from E1
    # # ==========================
    # if "Goldmine_E1_Emoji" in intraday.columns:
    #     gm_mask = intraday["Goldmine_E1_Emoji"] == "💰"

    #     if gm_mask.any():
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=intraday.loc[gm_mask, "Time"],
    #                 y=intraday.loc[gm_mask, "F_numeric"] + 120,  # tune offset
    #                 mode="text",
    #                 text=["💰"] * int(gm_mask.sum()),
    #                 textposition="middle center",
    #                 textfont=dict(size=18, color="gold"),
    #                 name="Goldmine E1",
    #                 hovertemplate=(
    #                     "Time: %{x}<br>"
    #                     "F%: %{y}<br>"
    #                     "Goldmine from Entry-1 💰 (+64F gain)"
    #                     "<extra></extra>"
    #                 ),
    #             ),
    #             row=1, col=1,
    #         )


    # ==========================
    # RVOL ♘
    # ==========================
    if "RVOL_5" in intraday.columns and "F_numeric" in intraday.columns:
        mask_rvol = pd.to_numeric(intraday["RVOL_5"], errors="coerce") > 1.2

        if mask_rvol.any():
            if "Kijun_F" in intraday.columns:
                mike  = pd.to_numeric(intraday["F_numeric"], errors="coerce")
                kijun = pd.to_numeric(intraday["Kijun_F"],   errors="coerce")
                above_mask = mask_rvol & (mike >= kijun)
                below_mask = mask_rvol & (mike <  kijun)
            else:
                above_mask = mask_rvol
                below_mask = pd.Series(False, index=intraday.index)

            for mask, pos_offset, textpos, color in [
                (above_mask, +50, "top center",    "#22c55e"),
                (below_mask, -50, "bottom center", "#ff3b3b"),
            ]:
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=intraday.loc[mask, "Time"],
                            y=intraday.loc[mask, "F_numeric"] + pos_offset,
                            mode="text",
                            text=["♘"] * int(mask.sum()),
                            textposition=textpos,
                            textfont=dict(size=26, color=color),
                            name="RVOL ♘",
                            showlegend=False,
                            customdata=intraday.loc[mask, "RVOL_5"].values,
                            hovertemplate=(
                                "Time: %{x}<br>"
                                "F%: %{y}<br>"
                                "♘ RVOL: %{customdata:.2f}<extra></extra>"
                            ),
                        ),
                        row=1, col=1,
                    )
    # ==========================
    # ♖ ROOK (TD line cross)
    # ==========================
    if "F_numeric" in intraday.columns:
        mike = pd.to_numeric(intraday["F_numeric"], errors="coerce")
        COOLDOWN_R = 10
        EPS_R      = 6
        last_r     = -999
        rook_hits  = []

        td_candidates = [
            ("TD Demand", "TD Demand Line F"),
            ("TD Supply", "TD Supply Line F"),
        ]

        for i in range(1, len(intraday)):
            m0, m1 = mike.iloc[i], mike.iloc[i - 1]
            if pd.isna(m0) or pd.isna(m1):
                continue
            hit = None
            for name, col in td_candidates:
                if col not in intraday.columns:
                    continue
                t0 = pd.to_numeric(intraday[col].iloc[i],     errors="coerce")
                t1 = pd.to_numeric(intraday[col].iloc[i - 1], errors="coerce")
                if pd.isna(t0) or pd.isna(t1):
                    continue
                prev_state = 1 if m1 > t1 + EPS_R else (-1 if m1 < t1 - EPS_R else 0)
                curr_state = 1 if m0 > t0 + EPS_R else (-1 if m0 < t0 - EPS_R else 0)
                if prev_state == 0 or curr_state == 0 or prev_state == curr_state:
                    continue
                hit = {"up_cross": prev_state == -1 and curr_state == 1, "label": f"{name} Cross"}
                break
            if hit is None:
                continue
            if i - last_r < COOLDOWN_R:
                continue
            rook_hits.append((i, hit))
            last_r = i

        for idx, hit in rook_hits:
            color   = "#22c55e" if hit["up_cross"] else "#ff3b3b"
            offset  = +50       if hit["up_cross"] else -50
            textpos = "top center" if hit["up_cross"] else "bottom center"
            fig.add_trace(
                go.Scatter(
                    x=[intraday["Time"].iloc[idx]],
                    y=[mike.iloc[idx] + offset],
                    mode="text",
                    text=["♖"],
                    textposition=textpos,
                    textfont=dict(size=26, color=color),
                    name="Rook ♖",
                    showlegend=False,
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "F%: %{y}<br>"
                        f"♖ {hit['label']}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

    # ==========================
    # ♙ PAWN (TD line seatbelt)
    # ==========================
    if "F_numeric" in intraday.columns:
        mike = pd.to_numeric(intraday["F_numeric"], errors="coerce")
        COOLDOWN_P = 10
        EPS_P      = 6
        last_p     = -999
        pawn_hits  = []

        for i in range(1, len(intraday)):
            m0, m1 = mike.iloc[i], mike.iloc[i - 1]
            if pd.isna(m0) or pd.isna(m1):
                continue
            hit = None
            for name, col in td_candidates:
                if col not in intraday.columns:
                    continue
                t0 = pd.to_numeric(intraday[col].iloc[i],     errors="coerce")
                t1 = pd.to_numeric(intraday[col].iloc[i - 1], errors="coerce")
                if pd.isna(t0) or pd.isna(t1):
                    continue
                prev_state = 1 if m1 > t1 + EPS_P else (-1 if m1 < t1 - EPS_P else 0)
                curr_state = 1 if m0 > t0 + EPS_P else (-1 if m0 < t0 - EPS_P else 0)
                if prev_state == 0 or curr_state == 0 or prev_state == curr_state:
                    continue
                hit = {"up_cross": prev_state == -1 and curr_state == 1, "label": f"{name} Seatbelt"}
                break
            if hit is None:
                continue
            if i - last_p < COOLDOWN_P:
                continue
            pawn_hits.append((i, hit))
            last_p = i

        for idx, hit in pawn_hits:
            color   = "#22c55e" if hit["up_cross"] else "#ff3b3b"
            offset  = +50       if hit["up_cross"] else -50
            textpos = "top center" if hit["up_cross"] else "bottom center"
            fig.add_trace(
                go.Scatter(
                    x=[intraday["Time"].iloc[idx]],
                    y=[mike.iloc[idx] + offset],
                    mode="text",
                    text=["♙"],
                    textposition=textpos,
                    textfont=dict(size=24, color=color),
                    name="Seatbelt ♙",
                    showlegend=False,
                    hovertemplate=(
                        "Time: %{x}<br>"
                        "F%: %{y}<br>"
                        f"♙ {hit['label']}<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

    # ==========================
    # 🔑 Z3 KEY MARKER
    # ==========================
    if "Z3_Key_Emoji" in intraday.columns:
        z3_mask = intraday["Z3_Key_Emoji"] == "🔑"
        if z3_mask.any():
            z3_row = intraday[z3_mask].iloc[0]
            z3_score = z3_row.get("Z3_Score", float("nan"))
            z3_side  = z3_row.get("Z3_Key_Side", "call")
            z3_f     = float(z3_row["F_numeric"])

            # Position: above Mike for call signal, below for put
            if "Kijun_F" in intraday.columns:
                kijun_val = pd.to_numeric(
                    intraday.loc[z3_mask, "Kijun_F"], errors="coerce"
                ).iloc[0]
                offset = 90 if z3_f >= kijun_val else -90
            else:
                offset = 90 if z3_side == "call" else -90

            score_str = f"{z3_score:.2f}" if not np.isnan(z3_score) else "n/a"
            fig.add_trace(
                go.Scatter(
                    x=intraday.loc[z3_mask, "Time"].iloc[:1],
                    y=[z3_f + offset],
                    mode="text",
                    text=["🔑"],
                    textposition="middle center",
                    textfont=dict(size=24),
                    name="Z3 Key 🔑",
                    showlegend=False,
                    hovertemplate=(
                        "Time: %{x}<br>"
                        f"F%: {z3_f:.0f}<br>"
                        f"Z3: {score_str}<br>"
                        f"Side: {z3_side}<br>"
                        "Z3 Key — favorable ignition after MAE"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

         # Nose (dominant TPO)
    poc_f_level = None
    nose_time = None

    nose_row = mp_df[mp_df["👃🏽"] == "👃🏽"]

    if not nose_row.empty:
        poc_f_level = int(nose_row["F% Level"].iat[0])
        nose_time = nose_row["Time"].iat[0]

        fig.add_trace(
            go.Scatter(
                x=[intraday["Time"].iat[-1]],
                y=[poc_f_level],
                mode="text",
                text=["👃🏽"],
                textposition="middle right",
                textfont=dict(size=18),
                hovertemplate=(
                    "👃🏽 Nose POC<br>"
                    f"F% Level: {poc_f_level}<br>"
                    f"First seen: {nose_time}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # Ear (older dominant %Vol)
        ear_row = mp_df[mp_df["🦻🏼"] == "🦻🏼"].sort_values(by="%Vol", ascending=False).head(1)
        if not ear_row.empty:
            ear_f_level = int(ear_row["F% Level"].iat[0])
            ear_vol = float(ear_row["%Vol"].iat[0])
            ear_time = ear_row["Time"].iat[0]

            fig.add_hline(
                y=ear_f_level,
                line_color="darkgray",
                line_dash="dot",
                line_width=1.2,
                row=1,
                col=1,
                annotation_text="🦻🏼 Ear Vol POC",
                annotation_position="bottom right",
                annotation_font_color="gray",
                showlegend=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=[intraday["Time"].iat[-1]],
                    y=[ear_f_level],
                    mode="text",
                    text=["🦻🏼"],
                    textposition="middle right",
                    textfont=dict(size=18),
                    hovertemplate=(
                        "🦻🏼 Ear Vol POC<br>"
                        f"%Vol: {ear_vol:.2f}<br>"
                        f"First seen: {ear_time}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.update_layout(
        height=800,
        showlegend=False,
    )

    return fig


def run_ticker_analysis(
    ticker: str,
    start_date: date,
    end_date: date,
    timeframe: str,
    gap_threshold_decimal: float,
):
    """Full pipeline for a single ticker: data, signals, profile, charts and tables."""
    # 1) Previous daily bar
    prev = fetch_prev_daily(ticker, start_date)
    prev_open = prev["open"]
    prev_high = prev["high"]
    prev_low = prev["low"]
    prev_close = prev["close"]
    prev_range = prev["range"]

    prev_close_str = f"{prev_close:.2f}" if prev_close is not None else "N/A"
    prev_high_str = f"{prev_high:.2f}" if prev_high is not None else "N/A"
    prev_low_str = f"{prev_low:.2f}" if prev_low is not None else "N/A"
    prev_range_str = f"{prev_range:.2f}" if prev_range is not None else "N/A"

    # c1, c2, c3, c4 = st.columns(4)
    # c1.metric("Prev Close", prev_close_str)
    # c2.metric("Prev High", prev_high_str)
    # c3.metric("Prev Low", prev_low_str)
    # c4.metric("Yesterday Range", prev_range_str)

    # 2) Intraday data
    intraday = fetch_intraday(ticker, start_date, end_date, timeframe)
    if intraday.empty:
        st.error(f"No intraday data for {ticker}.")
        return None

    # Last price
    last_price = float(intraday["Close"].iloc[-1])
    last_price_str = f"{last_price:.2f}"
    st.header(f"📈 {ticker} · {last_price_str}")

    # 3) F metrics
    intraday = calculate_f_numeric(intraday, prev_close)
    intraday = calculate_f_percentage(intraday, prev_close)
    # 5b) RVOL Alerts (Extreme / Strong / Moderate)
    intraday = apply_rvol_alerts(intraday, rvol_col="RVOL_5")
    # --- Ichimoku in F-space (Mike axis) ---
    intraday = apply_ichimoku_f_levels(
        intraday,
        prev_close=prev_close,
        tenkan_period=9,
        kijun_period=26,
    )


    # STD expansion (🐦‍🔥) on F_numeric
    intraday = apply_std_expansion(
        intraday,
        window=9,
        anchor_lookback=5,
    )


    # 5) RVOL
    intraday = add_rvol(intraday, window=5)

    # 6) Bollinger stuff in F-space (Mike)
    intraday = apply_bollinger_suite(
        intraday,
        window=20,
        scale_factor=10,
        tight_window=5,
        percentile_threshold=10,
        anchor_lookback=5,
        rvol_threshold=1.2,
    )


    # 6) Advanced TD and MIDAS
    intraday = apply_td_advanced_signals(intraday)
    intraday = compute_midas_curves(
        intraday,
        price_col="F_numeric",
        volume_col="Volume",
    )
   # 10) Physics core (vector %, momentum, capacitance, vol composite)
    intraday = apply_physics_core(
        intraday,
        rvol_col="RVOL_5",
        range_col="Range",
    )
    # 7) Range
    if "High" in intraday.columns and "Low" in intraday.columns:
        intraday["Range"] = intraday["High"] - intraday["Low"]

    # 8) Initial Balance
    ib_stats = compute_initial_balance(intraday, bars=12)

    # 9) Market Profile (TPO) from component
    mp_df, ib_info = compute_market_profile(
        intraday,
        price_col="F_numeric",
    )

 # 11) Mike Entry System (🎯 / 🎯2 / 🎯3 with physics filters)
    intraday = apply_entry_system(
        intraday,
        ib_info=ib_info,
        use_physics=True,
    )

    intraday = apply_T0_door(intraday, band_distance=5)
    intraday = apply_T1_horse(intraday)
    intraday = apply_T2_lightning(intraday)
    intraday = apply_parallel_phase(intraday)
    intraday = apply_goldmine_e1(intraday, dist=64)

    intraday = apply_e1_kijun_evil_eye(intraday)

    # Z3 momentum score + Key signal (🔑)
    intraday = apply_z3_key(intraday, threshold=1.5)






    # 10) Simple letter profile for tails
    letter_profile_df = build_letter_profile(intraday, mike_col="F_numeric")

    # Session date string for filenames (e.g. 2025-11-01)
    if "Date" in intraday.columns and len(intraday) > 0:
        d0 = intraday["Date"].iloc[0]
        session_date_str = d0.strftime("%Y-%m-%d") if hasattr(d0, "strftime") else str(d0)
    else:
        session_date_str = str(start_date)


    # ==============================
    # MARKET PROFILE (TPO)
    # ==============================
    with st.expander("🏛 Market Profile (TPO)", expanded=False):
        # IB metrics from compute_market_profile's ib_info (if present)
        ib_high_val = ib_info.get("IB_High")
        ib_low_val = ib_info.get("IB_Low")
        ib_mid_val = ib_info.get("IB_Mid_Third")
        ib_top_val = ib_info.get("IB_Top_Third")

        col_ib1, col_ib2, col_ib3, col_ib4 = st.columns(4)
        col_ib1.metric(
            "IB High (F%)",
            f"{ib_high_val:.1f}" if ib_high_val is not None and not pd.isna(ib_high_val) else "N/A",
        )
        col_ib2.metric(
            "IB Low (F%)",
            f"{ib_low_val:.1f}" if ib_low_val is not None and not pd.isna(ib_low_val) else "N/A",
        )
        col_ib3.metric(
            "IB Mid Third",
            f"{ib_mid_val:.1f}" if ib_mid_val is not None and not pd.isna(ib_mid_val) else "N/A",
        )
        col_ib4.metric(
            "IB Top Third",
            f"{ib_top_val:.1f}" if ib_top_val is not None and not pd.isna(ib_top_val) else "N/A",
        )

        if not mp_df.empty:
            # Only show columns that exist (avoid KeyErrors)
            desired_cols = [
                "F% Level",
                "Letters",
                "TPO_Count",
                "%Vol",
                "Tail",
                "✅ ValueArea",
                "🦻🏼",
                "👃🏽",
            ]
            show_cols = [c for c in desired_cols if c in mp_df.columns]
            st.dataframe(
                mp_df[show_cols],
                use_container_width=True,
            )
        else:
            st.info("Market Profile dataframe is empty.")

    # ==============================
    # MAIN INTRADAY TABLE
    # ==============================
    # with st.expander("📋 Intraday Table", expanded=False):
    #     base_cols = [
    #         "Date",
    #         "Time",
    #         "Volume",
    #         "RVOL_5",
    #         "F_numeric",
    #         "F%",
    #         "Close",
    #         "Range",
    #         "Low of Day",
    #         "High of Day",
    #     ]
    #     show_cols = [c for c in base_cols if c in intraday.columns]

    #     st.dataframe(
    #         intraday[show_cols],
    #         use_container_width=True,
    #     )



      # ==============================
    # CHART
    # ==============================
    # Wrap main chart so JS can read ticker + date
    st.markdown(
        f'<div class="mike-main-chart" data-ticker="{ticker}" data-session="{session_date_str}">',
        unsafe_allow_html=True,
    )

    fig = build_chart(
        intraday=intraday,
        ib_stats=ib_stats,
        profile_df_letters=letter_profile_df,
        mp_df=mp_df,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{ticker}-{session_date_str}",  # camera icon name
                "scale": 3,
            }
        }
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # Return intraday so components can build JSONs, etc.
    return intraday, mp_df

# =============================
# MAIN — Run button
# =============================
# =============================
# MAIN — Run button
# =============================
gap_threshold, gap_threshold_decimal = get_gap_settings()

# When you click, just flip the flag ON
if st.sidebar.button("Run Analysis"):
    if not tickers:
        st.warning("Please select at least one ticker.")
        st.session_state.analysis_run = False
    else:
        st.session_state.analysis_run = True

# ============================================
# ALWAYS keep json_map INSIDE the run-block
# ============================================
if st.session_state.analysis_run and tickers:

    json_map = {}   # ← EXISTS HERE AND BELOW 🔥

    tabs = st.tabs([f"{t}" for t in tickers])
    for idx, tkr in enumerate(tickers):
        with tabs[idx]:
            result = run_ticker_analysis(
                ticker=tkr,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                gap_threshold_decimal=gap_threshold_decimal,
            )

            if result is None:
                continue

            intraday_df, mp_df = result

            # build JSON for this ticker if data exists
            if isinstance(intraday_df, pd.DataFrame) and not intraday_df.empty:
                json_map[tkr] = build_basic_json(intraday_df, tkr, mp_df)

    # -----------------------------------------
    # RENDER JSON DOWNLOAD BUTTON (INSIDE BLOCK)
    # -----------------------------------------
    render_json_batch_download(json_map)




    # -----------------------------------------
    # 📸 Download ALL Plotly charts (same as camera icon)
    # -----------------------------------------
    components.html(
        """
        <button id="dlAllChartsBtn"
            style="
                padding: 8px 14px;
                background: #4CAF50;
                color: white;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-size: 14px;
                margin-top: 10px;
            ">
            📸 Download ALL Charts (camera-quality)
        </button>

        <script>
        const btn = document.getElementById("dlAllChartsBtn");

        btn.onclick = async () => {
            // charts live in the parent Streamlit document
            const parentDoc = window.parent.document;
            const plots = parentDoc.querySelectorAll('.js-plotly-plot');

            if (!plots.length) {
                alert("No charts found to download.");
                return;
            }

                 // Loop through each chart and trigger Plotly's own downloader
            for (let i = 0; i < plots.length; i++) {
                const gd = plots[i];

                // Default name in case we can't read config
                let filename = `volmike_chart_${i + 1}`;

                // If Plotly config has a toImageButtonOptions.filename, reuse it
                if (
                    gd._context &&
                    gd._context.toImageButtonOptions &&
                    gd._context.toImageButtonOptions.filename
                ) {
                    filename = gd._context.toImageButtonOptions.filename;
                }

                // Plotly is loaded in the parent window
                if (window.parent.Plotly && window.parent.Plotly.downloadImage) {
                    await window.parent.Plotly.downloadImage(gd, {
                        format: "png",
                        filename: filename,
                        scale: 3,      // HD, same idea as toolbar
                    });
                }
            }

        };
        </script>
        """,
        height=80,
    )