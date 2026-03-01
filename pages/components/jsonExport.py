# pages/components/jsonExport.py

from __future__ import annotations

import io
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SECTOR_MAP: dict[str, list[str]] = {
    "ETFs":           ["spy", "qqq"],
    "finance":        ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
    "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
    "Software":       ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
    "Futures":        ["nq", "es", "gc", "ym", "cl"],
}

_Z3_COLS   = ("Z3_Score", "z3", "Z3", "Z3_score")
_RV_COLS   = ("RVOL_5", "RVOL", "rvol")
_TIME_COLS = ("Time", "time", "Datetime", "datetime", "Timestamp", "timestamp", "DateTime")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def detect_sector(ticker: str) -> str:
    t = ticker.lower()
    for sector, tickers in SECTOR_MAP.items():
        if t in tickers:
            return sector
    return "Other"


def human_volume(n) -> str:
    try:
        n = float(n)
    except Exception:
        return str(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(int(n))


def round_all_numeric(obj):
    """Recursively round floats to 2dp. Leaves bools untouched."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, dict):
        return {k: round_all_numeric(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_all_numeric(v) for v in obj]
    try:
        return round(float(obj), 2)
    except Exception:
        return obj


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series("", index=df.index)


def _fmt_time(val) -> str:
    try:
        return pd.to_datetime(val).strftime("%H:%M")
    except Exception:
        return str(val)


def _resolve_col(df: pd.DataFrame, candidates: tuple) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


# ─────────────────────────────────────────────────────────────
# PERIMETER WINDOW  (bishops + horses + z3)
# ─────────────────────────────────────────────────────────────

def _window_analysis(
    intraday: pd.DataFrame,
    center_pos: int,
    perimeter: int,
    f_series: pd.Series,
    rv_col: str | None,
    z3_col: str | None,
) -> dict:
    n     = len(intraday)
    start = max(0, center_pos - perimeter)
    end   = min(n - 1, center_pos + perimeter)

    pre_b  = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
    post_b = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
    pre_h:  list[float] = []
    post_h: list[float] = []

    rvol_col =  rv_col
    has_kijun_col = "Kijun_F" in intraday.columns

    for pos in range(start, end + 1):

        tb = pre_b  if pos < center_pos else post_b
        th = pre_h  if pos < center_pos else post_h

        # Yellow bishop — BBW Tight (🐝)
        if "BBW_Tight_Emoji" in intraday.columns:
            val = str(intraday["BBW_Tight_Emoji"].iat[pos])
            if val.strip() == "🐝":
                tb["yellow"] += 1

        # Purple bishop — STD Alert (🐦‍🔥)
        if "STD_Alert" in intraday.columns:
            val = str(intraday["STD_Alert"].iat[pos])
            if val.strip() not in ("", "nan"):
                tb["purple"] += 1

        # Green/Red bishop — BBW Expansion (🔥)
        if "BBW Alert" in intraday.columns:
            val = str(intraday["BBW Alert"].iat[pos])
            if val.strip() == "🔥":
                fv = f_series.iat[pos]
                kv = (
                    pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce")
                    if has_kijun_col else float("nan")
                )
                if pd.notna(fv) and pd.notna(kv):
                    if fv >= kv:
                        tb["green"] += 1
                    else:
                        tb["red"] += 1
                else:
                    tb["green"] += 1

        # Horses — RVOL > 1.2
        if rvol_col is not None:
            rv = pd.to_numeric(intraday[rvol_col].iat[pos], errors="coerce")
            if pd.notna(rv) and rv > 1.2:
                th.append(round(float(rv), 2))

    # Z3 window stats
    z3_on          = False
    z3_value       = None
    z3_last3       = None
    z3_max_time    = None
    z3_max_bars    = None
    time_col       = _resolve_col(intraday, _TIME_COLS)

    if z3_col is not None:
        lo = max(0, center_pos - perimeter)
        hi = min(n - 1, center_pos + perimeter)
        s  = pd.to_numeric(intraday[z3_col].iloc[lo: hi + 1], errors="coerce")

        if s.notna().any():
            peak_idx   = int(s.abs().idxmax())
            z3_value   = float(s.loc[peak_idx])
            z3_on      = abs(z3_value) >= 1.5
            z3_max_bars = peak_idx - center_pos
            if time_col is not None:
                z3_max_time = str(intraday[time_col].iat[peak_idx])

        s3 = pd.to_numeric(
            intraday[z3_col].iloc[center_pos: min(n - 1, center_pos + 2) + 1],
            errors="coerce",
        )
        if s3.notna().any():
            z3_last3 = float(s3.loc[int(s3.abs().idxmax())])

    return {
        "pre":  {
            "bishops": {k: v for k, v in pre_b.items()  if v > 0},
            "horses":  {"count": len(pre_h),  "rvolValues": pre_h},
        },
        "post": {
            "bishops": {k: v for k, v in post_b.items() if v > 0},
            "horses":  {"count": len(post_h), "rvolValues": post_h},
        },
        "z3On":               z3_on,
        "z3Value":            z3_value,
        "z3ValueLast3":       z3_last3,
        "z3MaxTime":          z3_max_time,
        "z3MaxBarsFromEntry": z3_max_bars,
    }


# ─────────────────────────────────────────────────────────────
# ENTRIES  (E1 / E2 / E3 / Reclaim / Blocked + exit)
# ─────────────────────────────────────────────────────────────

def extract_entries(intraday: pd.DataFrame, perimeter: int = 4) -> dict:
    if intraday is None or intraday.empty:
        return {"call": [], "put": []}

    call_entries: list[dict] = []
    put_entries:  list[dict] = []
    n = len(intraday)
    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")

    rv_col      = _resolve_col(intraday, _RV_COLS)
    z3_col      = _resolve_col(intraday, _Z3_COLS)

    # Exit signal index sets
    put_signals = (
        set(intraday.index[_col(intraday, "Put_FirstEntry_Emoji").isin(["🎯", "⏳"])])
        | set(intraday.index[_col(intraday, "Put_DeferredEntry_Emoji") == "🧿"])
    )
    call_signals = (
        set(intraday.index[_col(intraday, "Call_FirstEntry_Emoji").isin(["🎯", "⏳"])])
        | set(intraday.index[_col(intraday, "Call_DeferredEntry_Emoji") == "🧿"])
    )

    # Last bar fallback (3:55)
    last_idx   = intraday.index[-1]
    last_price = float(intraday.at[last_idx, "Close"])
    last_f     = float(intraday.at[last_idx, "F_numeric"])
    last_time  = _fmt_time(intraday.at[last_idx, "Time"])

    def _exit_reason(exit_idx) -> str:
        checks = [
            ("Put_FirstEntry_Emoji",    "🎯", "Put E1 🎯"),
            ("Put_FirstEntry_Emoji",    "⏳", "Put E1 ⏳"),
            ("Put_DeferredEntry_Emoji", "🧿", "Put 🧿"),
            ("Call_FirstEntry_Emoji",   "🎯", "Call E1 🎯"),
            ("Call_FirstEntry_Emoji",   "⏳", "Call E1 ⏳"),
            ("Call_DeferredEntry_Emoji","🧿", "Call 🧿"),
        ]
        for col_name, emoji, label in checks:
            if col_name in intraday.columns and intraday.at[exit_idx, col_name] == emoji:
                return label
        return "signal"

    def _find_exit(entry_idx, opposite: set) -> dict:
        entry_pos  = intraday.index.get_loc(entry_idx)
        candidates = [ix for ix in opposite if intraday.index.get_loc(ix) > entry_pos]
        if candidates:
            exit_idx = min(candidates, key=lambda ix: intraday.index.get_loc(ix))
            return {
                "time":   _fmt_time(intraday.at[exit_idx, "Time"]),
                "price":  float(intraday.at[exit_idx, "Close"]),
                "fLevel": float(intraday.at[exit_idx, "F_numeric"]),
                "reason": _exit_reason(exit_idx),
            }
        return {"time": last_time, "price": last_price, "fLevel": last_f, "reason": "close"}

    def _attach_exit(ex: dict, entry_price: float, entry_f: float) -> dict:
        ex["priceMoveUSD"] = round(ex["price"] - entry_price, 2)
        ex["fMove"]        = round(ex["fLevel"] - entry_f, 2)
        return ex

    def _perimeter(pos: int) -> dict:
        return _window_analysis(
            intraday, pos, perimeter, f,
            rv_col, z3_col,
        )

    def _add(target: list, label: str, idx,
             opposite: set | None = None,
             with_perimeter: bool = False,
             extra: dict | None = None):
        pos         = intraday.index.get_loc(idx)
        entry_price = float(intraday.at[idx, "Close"])
        entry_f     = float(intraday.at[idx, "F_numeric"])
        row: dict   = {
            "type":   label,
            "time":   _fmt_time(intraday.at[idx, "Time"]),
            "price":  entry_price,
            "fLevel": entry_f,
        }
        if extra:
            row.update(extra)
        if with_perimeter:
            row["perimeter"] = _perimeter(pos)
        if opposite is not None:
            ex = _find_exit(idx, opposite)
            row["exit"] = _attach_exit(ex, entry_price, entry_f)
        target.append(row)

    # PUT
    for i in intraday.index[_col(intraday, "Put_FirstEntry_Emoji") == "🎯"]:
        _add(put_entries, "Put E1 🎯", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_FirstEntry_Emoji") == "⏳"]:
        _add(put_entries, "Put E1 ⏳ Blocked", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_DeferredEntry_Emoji") == "🧿"]:
        horse = _col(intraday, "Put_DeferredReinforce_Emoji").at[i] == "❗️"
        _add(put_entries, "Put Reclaim 🧿", i,
             opposite=call_signals, with_perimeter=True, extra={"horse": horse})

    for i in intraday.index[_col(intraday, "Put_SecondEntry_Emoji") == "🎯2"]:
        _add(put_entries, "Put E2 🎯2", i, opposite=call_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Put_ThirdEntry_Emoji") == "🎯3"]:
        _add(put_entries, "Put E3 🎯3", i)

    # CALL
    for i in intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "🎯"]:
        _add(call_entries, "Call E1 🎯", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "⏳"]:
        _add(call_entries, "Call E1 ⏳ Blocked", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_DeferredEntry_Emoji") == "🧿"]:
        horse = _col(intraday, "Call_DeferredReinforce_Emoji").at[i] == "❗️"
        _add(call_entries, "Call Reclaim 🧿", i,
             opposite=put_signals, with_perimeter=True, extra={"horse": horse})

    for i in intraday.index[_col(intraday, "Call_SecondEntry_Emoji") == "🎯2"]:
        _add(call_entries, "Call E2 🎯2", i, opposite=put_signals, with_perimeter=True)

    for i in intraday.index[_col(intraday, "Call_ThirdEntry_Emoji") == "🎯3"]:
        _add(call_entries, "Call E3 🎯3", i)

    return {"call": call_entries, "put": put_entries}


# ─────────────────────────────────────────────────────────────
# EXPANSION INSIGHT
# ─────────────────────────────────────────────────────────────

def detect_expansion_near_e1(intraday: pd.DataFrame, perimeter: int = 10) -> dict:
    out = {
        "bbw": {"present": False, "time": None, "count": 0},
        "std": {"present": False, "time": None, "count": 0},
    }
    if intraday is None or intraday.empty:
        return out

    call_e1 = intraday.index[_col(intraday, "Call_FirstEntry_Emoji") == "🎯"]
    put_e1  = intraday.index[_col(intraday, "Put_FirstEntry_Emoji")  == "🎯"]
    if len(call_e1) == 0 and len(put_e1) == 0:
        return out

    e1_pos = min(intraday.index.get_loc(ix) for ix in list(call_e1) + list(put_e1))
    n      = len(intraday)
    start  = max(0, e1_pos - perimeter)
    end    = min(n - 1, e1_pos + perimeter)

    def _count(series: pd.Series, emoji: str) -> tuple[int, int]:
        before = after = 0
        for pos in range(start, end + 1):
            if series.iloc[pos] == emoji:
                if pos <= e1_pos:
                    before += 1
                else:
                    after += 1
        return before, after

    def _timing(b: int, a: int) -> str | None:
        if b and a:  return "both"
        if b:        return "before"
        if a:        return "after"
        return None

    bbw_s = intraday.get("BBW Alert")
    std_s = intraday.get("STD_Alert")

    if bbw_s is not None:
        b, a = _count(bbw_s, "🔥")
        if b + a:
            out["bbw"] = {"present": True, "time": _timing(b, a), "count": b + a}

    if std_s is not None:
        b, a = _count(std_s, "🐦\u200d🔥")
        if b + a:
            out["std"] = {"present": True, "time": _timing(b, a), "count": b + a}

    return out


# ─────────────────────────────────────────────────────────────
# MILESTONES
# ─────────────────────────────────────────────────────────────

def extract_milestones(intraday: pd.DataFrame) -> dict:
    def _first(emoji_col: str, emoji: str) -> dict:
        rows = intraday[_col(intraday, emoji_col) == emoji]
        if rows.empty:
            return {}
        r = rows.iloc[0]
        return {"Time": _fmt_time(r["Time"]), "Price": float(r["Close"]), "F%": float(r["F_numeric"])}

    goldmine = [
        {"Time": _fmt_time(r["Time"]), "Price": float(r["Close"]), "F%": float(r["F_numeric"])}
        for _, r in intraday[_col(intraday, "Goldmine_E1_Emoji") == "💰"].iterrows()
    ]
    return {
        "T0":       _first("T0_Emoji", "🚪"),
        "T1":       _first("T1_Emoji", "🏇🏼"),
        "T2":       _first("T2_Emoji", "⚡"),
        "goldmine": goldmine,
    }


# ─────────────────────────────────────────────────────────────
# MARKET PROFILE  (kept for compat; nose/ear excluded from JSON)
# ─────────────────────────────────────────────────────────────

def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
    if mp_df is None or mp_df.empty or "F% Level" not in mp_df.columns:
        return {}
    df = mp_df.copy()
    for c, default in [("TPO_Count", 0), ("%Vol", 0.0), ("🦻🏼", ""), ("👃🏽", "")]:
        if c not in df.columns:
            df[c] = default

    out: dict = {}

    nose_row = df[df["👃🏽"] == "👃🏽"]
    if nose_row.empty:
        nose_row = df.sort_values("TPO_Count", ascending=False).head(1)
    if not nose_row.empty:
        out["nose"] = {"fLevel": int(nose_row["F% Level"].iloc[0]),
                       "tpoCount": int(nose_row["TPO_Count"].iloc[0])}

    ear_row = df[df["🦻🏼"] == "🦻🏼"]
    if ear_row.empty:
        ear_row = df.sort_values("%Vol", ascending=False).head(1)
    if not ear_row.empty:
        out["ear"] = {"fLevel": int(ear_row["F% Level"].iloc[0]),
                      "percentVol": float(ear_row["%Vol"].iloc[0])}

    return out


# ─────────────────────────────────────────────────────────────
# BACKWARD-COMPAT STUBS  (imported by other modules)
# ─────────────────────────────────────────────────────────────

def extract_vector_capacitance(intraday: pd.DataFrame, perimeter: int = 5) -> dict:
    return {"call": {}, "put": {}}


def extract_profile_cross_insight(
    intraday: pd.DataFrame,
    mp_block: dict | None,
    goldmine_dist: float = 64.0,
) -> dict:
    return {}


# ─────────────────────────────────────────────────────────────
# RANGE EXTENSION
# ─────────────────────────────────────────────────────────────

def extract_range_extension(
    intraday: pd.DataFrame,
    ib_high_f: float | None,
    ib_low_f:  float | None,
    perimeter: int = 4,
) -> dict:
    if intraday is None or intraday.empty or ib_high_f is None or ib_low_f is None:
        return {}

    f      = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    n      = len(intraday)
    IB_END = 12

    rv_col      = _resolve_col(intraday, _RV_COLS)
    z3_col      = _resolve_col(intraday, _Z3_COLS)
    rvol_col    = next((c for c in ("RVOL_5", "RVOL", "rvol") if c in intraday.columns), None)
    has_kijun_col = "Kijun_F" in intraday.columns

    def _win(start: int, end: int) -> dict:
        bishops: dict[str, int] = {"yellow": 0, "purple": 0, "green": 0, "red": 0}
        horses:  list[float]    = []

        for pos in range(max(0, start), min(n - 1, end) + 1):
            # Yellow bishop — BBW Tight (🐝)
            if "BBW_Tight_Emoji" in intraday.columns:
                if str(intraday["BBW_Tight_Emoji"].iat[pos]).strip() == "🐝":
                    bishops["yellow"] += 1

            # Purple bishop — STD Alert (🐦‍🔥)
            if "STD_Alert" in intraday.columns:
                if str(intraday["STD_Alert"].iat[pos]).strip() not in ("", "nan"):
                    bishops["purple"] += 1

            # Green/Red bishop — BBW Expansion (🔥)
            if "BBW Alert" in intraday.columns:
                if str(intraday["BBW Alert"].iat[pos]).strip() == "🔥":
                    fv = f.iat[pos]
                    kv = (
                        pd.to_numeric(intraday["Kijun_F"].iat[pos], errors="coerce")
                        if has_kijun_col else float("nan")
                    )
                    if pd.notna(fv) and pd.notna(kv):
                        if fv >= kv:
                            bishops["green"] += 1
                        else:
                            bishops["red"] += 1
                    else:
                        bishops["green"] += 1

            # Horses — RVOL > 1.2
            if rvol_col is not None:
                rv = pd.to_numeric(intraday[rvol_col].iat[pos], errors="coerce")
                if pd.notna(rv) and rv > 1.2:
                    horses.append(round(float(rv), 2))

        return {
            "bishops": {k: v for k, v in bishops.items() if v > 0},
            "horses":  {"count": len(horses), "rvolValues": horses},
        }

    def _scan(direction: str) -> dict:
        ext_loc = None
        for i in range(IB_END, n):
            fv = f.iat[i]
            if pd.notna(fv):
                if direction == "high" and fv > ib_high_f:
                    ext_loc = i
                    break
                elif direction == "low" and fv < ib_low_f:
                    ext_loc = i
                    break
        if ext_loc is None:
            return {}

        pre_start = max(IB_END, ext_loc - perimeter)
        post_end  = min(n - 1,  ext_loc + perimeter)

        z3_val: float | None = None
        if z3_col is not None:
            raw = pd.to_numeric(intraday[z3_col].iat[ext_loc], errors="coerce")
            if pd.notna(raw):
                z3_val = float(raw)
        z3_on = z3_val is not None and abs(z3_val) >= 1.5

        time_str = None
        if "Time" in intraday.columns:
            time_str = _fmt_time(intraday["Time"].iat[ext_loc])

        return {
            "time":    time_str,
            "fLevel":  round(float(f.iat[ext_loc]), 2),
            "z3On":    bool(z3_on),
            "z3Value": round(z3_val, 2) if z3_val is not None else None,
            "pre":     _win(pre_start, ext_loc - 1),
            "post":    _win(ext_loc + 1, post_end),
        }

    result: dict = {}
    h = _scan("high")
    l = _scan("low")
    if h:
        result["aboveIBHigh"] = h
    if l:
        result["belowIBLow"] = l
    return result


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────

def build_basic_json(
    intraday: pd.DataFrame,
    ticker: str,
    mp_df: pd.DataFrame | None = None,
) -> dict:
    if intraday is None or intraday.empty:
        return {}

    total_vol     = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
    last_date     = intraday["Date"].iloc[-1]     if "Date"   in intraday.columns else date.today()
    sector        = detect_sector(ticker)
    slug          = f"{ticker.lower()}-{last_date}-{sector}"
    open_price    = float(intraday["Open"].iloc[0])   if "Open"  in intraday.columns else None
    close_price   = float(intraday["Close"].iloc[-1]) if "Close" in intraday.columns else None

    # MIDAS Bear
    try:
        bear_idx         = intraday["MIDAS_Bear"].first_valid_index()
        midas_bear_time  = intraday.loc[bear_idx, "Time"]            if bear_idx else None
        midas_bear_f     = float(intraday.loc[bear_idx, "F_numeric"]) if bear_idx else None
        midas_bear_price = float(intraday.loc[bear_idx, "Close"])     if bear_idx else None
    except Exception:
        midas_bear_time = midas_bear_f = midas_bear_price = None

    # MIDAS Bull
    try:
        bull_idx         = intraday["MIDAS_Bull"].first_valid_index()
        midas_bull_time  = intraday.loc[bull_idx, "Time"]            if bull_idx else None
        midas_bull_f     = float(intraday.loc[bull_idx, "F_numeric"]) if bull_idx else None
        midas_bull_price = float(intraday.loc[bull_idx, "Close"])     if bull_idx else None
    except Exception:
        midas_bull_time = midas_bull_f = midas_bull_price = None

    # Initial Balance (first 12 bars)
    try:
        ib_slice  = intraday.iloc[:12]
        ib_high_f = float(ib_slice["F_numeric"].max())
        ib_low_f  = float(ib_slice["F_numeric"].min())

        ib_high_row   = intraday.loc[intraday["F_numeric"] == ib_high_f].iloc[0]
        ib_high_time  = ib_high_row["Time"]
        ib_high_price = float(ib_high_row["Close"])

        ib_low_row    = intraday.loc[intraday["F_numeric"] == ib_low_f].iloc[0]
        ib_low_time   = ib_low_row["Time"]
        ib_low_price  = float(ib_low_row["Close"])
    except Exception:
        ib_high_f = ib_low_f = None
        ib_high_time = ib_low_time = None
        ib_high_price = ib_low_price = None

    mp_block = extract_market_profile(mp_df)

    payload = {
        "name":             str(ticker).lower(),
        "date":             str(last_date),
        "sector":           sector,
        "slug":             slug,
        "totalVolume":      human_volume(total_vol),
        "open":             open_price,
        "close":            close_price,
        "expansionInsight": detect_expansion_near_e1(intraday, perimeter=9),
        "entries":          extract_entries(intraday, perimeter=9),
        "milestones":       extract_milestones(intraday),
        "marketProfile":    {k: v for k, v in mp_block.items() if k not in ("nose", "ear")},
        "rangeExtension":   extract_range_extension(intraday, ib_high_f, ib_low_f, perimeter=9),
        "initialBalance": {
            "high": {"time": ib_high_time, "fLevel": ib_high_f, "price": ib_high_price},
            "low":  {"time": ib_low_time,  "fLevel": ib_low_f,  "price": ib_low_price},
        },
        "midas": {
            "bear": {"anchorTime": midas_bear_time,  "price": midas_bear_price,  "fLevel": midas_bear_f},
            "bull": {"anchorTime": midas_bull_time,  "price": midas_bull_price,  "fLevel": midas_bull_f},
        },
    }

    return round_all_numeric(payload)


# ─────────────────────────────────────────────────────────────
# DOWNLOAD WIDGET
# ─────────────────────────────────────────────────────────────

def render_json_batch_download(json_map: dict) -> None:
    if not json_map:
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for tkr, payload in json_map.items():
            safe  = payload.get("name", str(tkr)).lower()
            d     = payload.get("date", "")
            fname = f"{safe}-{d}.json" if d else f"{safe}.json"
            zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False))

    buffer.seek(0)
    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )