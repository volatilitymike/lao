# pages/components/jsonExport.py

import io
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st



SECTOR_MAP = {
    "ETFs": ["spy", "qqq"],
    "finance": ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
    "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
    "Software": ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
    "Futures": ["nq", "es", "gc", "ym", "cl"],
}
def detect_sector(ticker: str) -> str:
    t = ticker.lower()
    for sector, listing in SECTOR_MAP.items():
        if t in listing:
            return sector
    return "Other"

def human_volume(n):
    try:
        n = float(n)
    except:
        return n

    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.2f}K"
    else:
        return str(int(n))
def extract_entries(intraday: pd.DataFrame) -> dict:
    """
    Simple, stable extractor for Call/Put 🎯 entries.
    No prototypes, suffixes, milestones — clean and safe.
    """
    call_entries = []
    put_entries = []

    def add_entry(target, label, idx):
        target.append({
            "Type": label,
            "Time": pd.to_datetime(intraday.at[idx, "Time"]).strftime("%H:%M"),
            "Price": float(intraday.at[idx, "Close"]),
            "F%": float(intraday.at[idx, "F_numeric"]),
        })

    # PUTS
    for i in intraday.index[intraday["Put_FirstEntry_Emoji"] == "🎯"]:
        add_entry(put_entries, "Put 🎯1", i)
    for i in intraday.index[intraday["Put_SecondEntry_Emoji"] == "🎯2"]:
        add_entry(put_entries, "Put 🎯2", i)
    for i in intraday.index[intraday["Put_ThirdEntry_Emoji"] == "🎯3"]:
        add_entry(put_entries, "Put 🎯3", i)

    # CALLS
    for i in intraday.index[intraday["Call_FirstEntry_Emoji"] == "🎯"]:
        add_entry(call_entries, "Call 🎯1", i)
    for i in intraday.index[intraday["Call_SecondEntry_Emoji"] == "🎯2"]:
        add_entry(call_entries, "Call 🎯2", i)
    for i in intraday.index[intraday["Call_ThirdEntry_Emoji"] == "🎯3"]:
        add_entry(call_entries, "Call 🎯3", i)

    return {
        "call": call_entries,
        "put": put_entries,
    }



def detect_expansion_near_e1(
    intraday: pd.DataFrame,
    perimeter: int = 10
) -> dict:
    """
    Detect if BBW Alert (🔥) or STD Alert (🐦‍🔥) happened
    within +/- perimeter bars around Entry 1.

    Returns:
      {
        "bbw": {
          "present": True/False,
          "time": "before" / "after" / "both" / None,
          "count": int
        },
        "std": {
          "present": True/False,
          "time": "before" / "after" / "both" / None,
          "count": int
        }
      }

    NOTE:
      - "before" = at least one alert at or before E1, none after
      - "after"  = at least one alert after E1, none before
      - "both"   = alerts on both sides of E1
      - Bar == E1 is counted as "before" (change <= to < if you want strict).
    """

    out = {
        "bbw": {"present": False, "time": None, "count": 0},
        "std": {"present": False, "time": None, "count": 0},
    }

    if intraday is None or intraday.empty:
        return out

    # ---- Find Entry 1 (earliest of call/put) ----
    call_e1_idx = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1_idx  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    if len(call_e1_idx) == 0 and len(put_e1_idx) == 0:
        return out

    all_e1_idx = list(call_e1_idx) + list(put_e1_idx)
    # earliest by positional location
    e1_pos = min(intraday.index.get_loc(idx) for idx in all_e1_idx)

    # ---- Define perimeter window (positional) ----
    n = len(intraday)
    start = max(0, e1_pos - perimeter)
    end   = min(n - 1, e1_pos + perimeter)

    # Pre-grab alert series (if missing, there is no alert at all)
    bbw_series = intraday.get("BBW Alert")
    std_series = intraday.get("STD_Alert")

    # ---------- BBW 🔥 ----------
    if bbw_series is not None:
        bbw_before = 0
        bbw_after = 0

        for pos in range(start, end + 1):
            val = bbw_series.iloc[pos]
            if val == "🔥":
                # count bar on E1 as "before" (<=). Change to < if you prefer.
                if pos <= e1_pos:
                    bbw_before += 1
                else:
                    bbw_after += 1

        total = bbw_before + bbw_after
        if total > 0:
            out["bbw"]["present"] = True
            out["bbw"]["count"] = total

            if bbw_before > 0 and bbw_after == 0:
                out["bbw"]["time"] = "before"
            elif bbw_after > 0 and bbw_before == 0:
                out["bbw"]["time"] = "after"
            elif bbw_before > 0 and bbw_after > 0:
                out["bbw"]["time"] = "both"

    # ---------- STD 🐦‍🔥 ----------
    if std_series is not None:
        std_before = 0
        std_after = 0

        for pos in range(start, end + 1):
            val = std_series.iloc[pos]
            if val == "🐦‍🔥":
                if pos <= e1_pos:
                    std_before += 1
                else:
                    std_after += 1

        total = std_before + std_after
        if total > 0:
            out["std"]["present"] = True
            out["std"]["count"] = total

            if std_before > 0 and std_after == 0:
                out["std"]["time"] = "before"
            elif std_after > 0 and std_before == 0:
                out["std"]["time"] = "after"
            elif std_before > 0 and std_after > 0:
                out["std"]["time"] = "both"

    return out


def extract_milestones(intraday: pd.DataFrame) -> dict:
    """
    Extracts T0, T1, T2 and Goldmine_E1 hits into clean JSON-friendly dicts.
    - Missing values return {} (Mongo-safe)
    - Goldmine returns a list of hits
    """

    # --------------- T0 ----------------
    t0 = intraday[intraday.get("T0_Emoji", "") == "🚪"]
    if len(t0) > 0:
        row = t0.iloc[0]
        T0 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T0 = {}

    # --------------- T1 ----------------
    t1 = intraday[intraday.get("T1_Emoji", "") == "🏇🏼"]
    if len(t1) > 0:
        row = t1.iloc[0]
        T1 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T1 = {}

    # --------------- T2 ----------------
    t2 = intraday[intraday.get("T2_Emoji", "") == "⚡"]
    if len(t2) > 0:
        row = t2.iloc[0]
        T2 = {
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        }
    else:
        T2 = {}

    # --------------- GOLDMINE (E1 ladder) ----------------
    gm_hits = intraday[intraday.get("Goldmine_E1_Emoji", "") == "💰"]
    goldmine = []
    for _, row in gm_hits.iterrows():
        goldmine.append({
            "Time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
            "Price": float(row["Close"]),
            "F%": float(row["F_numeric"]),
        })

    return {
        "T0": T0,
        "T1": T1,
        "T2": T2,
        "goldmine": goldmine
    }



def extract_vector_capacitance(intraday: pd.DataFrame, perimeter: int = 5) -> dict:
    """
    Returns highest Vector_Capacitance BEFORE/AFTER Entry-1,
    separately for Call and Put.
    """

    if intraday.empty or "Vector_Capacitance" not in intraday.columns:
        return {"call": {}, "put": {}}

    def side_block(side: str, e1_idx):
        """Compute before/after for a specific entry index."""
        if e1_idx is None:
            return {}

        e1_loc = intraday.index.get_loc(e1_idx)

        start_before = max(0, e1_loc - perimeter)
        end_before   = e1_loc - 1

        start_after  = e1_loc + 1
        end_after    = min(len(intraday) - 1, e1_loc + perimeter)

        before_slice = intraday.iloc[start_before:end_before+1]
        after_slice  = intraday.iloc[start_after:end_after+1]

        # --- highest BEFORE ---
        before = {}
        if not before_slice.empty:
            temp = before_slice.dropna(subset=["Vector_Capacitance"])
            if not temp.empty:
                row = temp.loc[temp["Vector_Capacitance"].idxmax()]
                before = {
                    "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
                    "value": float(row["Vector_Capacitance"])
                }

        # --- highest AFTER ---
        after = {}
        if not after_slice.empty:
            temp = after_slice.dropna(subset=["Vector_Capacitance"])
            if not temp.empty:
                row = temp.loc[temp["Vector_Capacitance"].idxmax()]
                after = {
                    "time": pd.to_datetime(row["Time"]).strftime("%H:%M"),
                    "value": float(row["Vector_Capacitance"])
                }

        return {"before": before, "after": after}

    # Find entries for each side
    call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    call_idx = call_e1[0] if len(call_e1) > 0 else None
    put_idx  = put_e1[0]  if len(put_e1)  > 0 else None

    return {
        "call": side_block("call", call_idx),
        "put":  side_block("put",  put_idx)
    }
def extract_market_profile(mp_df: pd.DataFrame | None) -> dict:
    """
    Compact Nose/Ear block from Market Profile df.

    Returns:
      {
        "nose": {"fLevel": int, "tpoCount": int},
        "ear":  {"fLevel": int, "percentVol": float}
      }

    If mp_df is None/empty → {}.
    """
    if mp_df is None or mp_df.empty:
        return {}

    df = mp_df.copy()

    # Make sure columns exist
    if "F% Level" not in df.columns:
        return {}

    if "TPO_Count" not in df.columns:
        df["TPO_Count"] = 0
    if "%Vol" not in df.columns:
        df["%Vol"] = 0.0
    if "🦻🏼" not in df.columns:
        df["🦻🏼"] = ""
    if "👃🏽" not in df.columns:
        df["👃🏽"] = ""

    out: dict = {}

    # 👉 Nose (time POC)
    nose_row = df[df["👃🏽"] == "👃🏽"]
    if nose_row.empty:
        # fallback: max TPO_Count
        nose_row = df.sort_values(by="TPO_Count", ascending=False).head(1)

    if not nose_row.empty:
        out["nose"] = {
            "fLevel": int(nose_row["F% Level"].iloc[0]),
            "tpoCount": int(nose_row["TPO_Count"].iloc[0]),
        }

    # 👉 Ear (volume POC)
    ear_row = df[df["🦻🏼"] == "🦻🏼"]
    if ear_row.empty:
        # fallback: max %Vol
        ear_row = df.sort_values(by="%Vol", ascending=False).head(1)

    if not ear_row.empty:
        out["ear"] = {
            "fLevel": int(ear_row["F% Level"].iloc[0]),
            "percentVol": float(ear_row["%Vol"].iloc[0]),
        }

    return out
def extract_profile_cross_insight(
    intraday: pd.DataFrame,
    mp_block: dict | None,
    goldmine_dist: float = 64.0,
) -> dict:
    """
    Insight: did crossing the Market Profile level (Nose/Ear) in favor
    of the trade actually pay?

    For each level (Nose / Ear) we compute:

      call:
        - first Call 🎯1
        - if entry F < levelF → look for first bar where F >= levelF
        - after that cross, measure best F up-move from entry
        - check if move >= +goldmine_dist

      put:
        - first Put 🎯1
        - if entry F > levelF → look for first bar where F <= levelF
        - after that cross, measure best F down-move from entry
        - check if move >= goldmine_dist in favor of put

    Returns:
      {
        "nose": { "call": {...}, "put": {...} },
        "ear":  { "call": {...}, "put": {...} }
      }
    """
    if intraday is None or intraday.empty or not mp_block:
        return {}

    nose_info = mp_block.get("nose") or {}
    ear_info  = mp_block.get("ear")  or {}

    nose_f = nose_info.get("fLevel")
    ear_f  = ear_info.get("fLevel")

    if "F_numeric" not in intraday.columns or "Time" not in intraday.columns:
        return {}

    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")
    t = pd.to_datetime(intraday["Time"], format="%I:%M %p", errors="coerce")

    def _compute_for_level(level_f: float | None) -> dict:
        """Return {'call': {...}, 'put': {...}} for a single F level."""
        if level_f is None:
            return {"call": {}, "put": {}}

        # ---------- CALL SIDE ----------
        def _side_call():
            call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
            if len(call_e1) == 0:
                return {}

            idx = call_e1[0]
            loc = intraday.index.get_loc(idx)
            entry_f = f.iloc[loc]
            if pd.isna(entry_f):
                return {}

            entry_time = (
                t.iloc[loc].strftime("%H:%M")
                if pd.notna(t.iloc[loc])
                else intraday["Time"].iloc[loc]
            )

            if entry_f < level_f:
                pos = "below"
            elif entry_f > level_f:
                pos = "above"
            else:
                pos = "on"

            crossed = "no"
            cross_time = None
            cross_f = None
            best_after = None
            best_move = None
            goldmine_flag = "no"

            # We care about bullish cross-up if entry is below level
            if entry_f < level_f:
                after = f.iloc[loc + 1 :]
                hit_mask = after >= level_f
                if hit_mask.any():
                    hit_idx = hit_mask[hit_mask].index[0]
                    hit_loc = intraday.index.get_loc(hit_idx)

                    cross_f_val = f.iloc[hit_loc]
                    cross_ts = t.iloc[hit_loc]
                    cross_time = (
                        cross_ts.strftime("%H:%M")
                        if pd.notna(cross_ts)
                        else intraday["Time"].iloc[hit_loc]
                    )
                    cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
                    crossed = "yes"

                    future = f.iloc[hit_loc:]
                    best_after_val = future.max()
                    if pd.notna(best_after_val):
                        best_after = float(best_after_val)
                        best_move = float(best_after - entry_f)
                        if best_move >= goldmine_dist:
                            goldmine_flag = "yes"

            return {
                "entryF": float(entry_f),
                "entryTime": entry_time,
                "levelF": float(level_f),
                "entryPositionVsLevel": pos,
                "crossedInFavor": crossed,          # "yes"/"no"
                "crossTime": cross_time,
                "crossF": cross_f,
                "bestFAfterCross": best_after,
                "bestMoveFromEntry": best_move,
                "goldmineLike64F": goldmine_flag,   # "yes"/"no"
            }

        # ---------- PUT SIDE ----------
        def _side_put():
            put_e1 = intraday.index[intraday.get("Put_FirstEntry_Emoji", "") == "🎯"]
            if len(put_e1) == 0:
                return {}

            idx = put_e1[0]
            loc = intraday.index.get_loc(idx)
            entry_f = f.iloc[loc]
            if pd.isna(entry_f):
                return {}

            entry_time = (
                t.iloc[loc].strftime("%H:%M")
                if pd.notna(t.iloc[loc])
                else intraday["Time"].iloc[loc]
            )

            if entry_f > level_f:
                pos = "above"
            elif entry_f < level_f:
                pos = "below"
            else:
                pos = "on"

            crossed = "no"
            cross_time = None
            cross_f = None
            best_after = None
            best_move = None
            goldmine_flag = "no"

            # Bearish cross-down if entry is above level
            if entry_f > level_f:
                after = f.iloc[loc + 1 :]
                hit_mask = after <= level_f
                if hit_mask.any():
                    hit_idx = hit_mask[hit_mask].index[0]
                    hit_loc = intraday.index.get_loc(hit_idx)

                    cross_f_val = f.iloc[hit_loc]
                    cross_ts = t.iloc[hit_loc]
                    cross_time = (
                        cross_ts.strftime("%H:%M")
                        if pd.notna(cross_ts)
                        else intraday["Time"].iloc[hit_loc]
                    )
                    cross_f = float(cross_f_val) if pd.notna(cross_f_val) else None
                    crossed = "yes"

                    future = f.iloc[hit_loc:]
                    worst_after = future.min()
                    if pd.notna(worst_after):
                        best_after = float(worst_after)
                        best_move = float(entry_f - worst_after)
                        if best_move >= goldmine_dist:
                            goldmine_flag = "yes"

            return {
                "entryF": float(entry_f),
                "entryTime": entry_time,
                "levelF": float(level_f),
                "entryPositionVsLevel": pos,
                "crossedInFavor": crossed,
                "crossTime": cross_time,
                "crossF": cross_f,
                "bestFAfterCross": best_after,
                "bestMoveFromEntry": best_move,
                "goldmineLike64F": goldmine_flag,
            }

        return {
            "call": _side_call(),
            "put": _side_put(),
        }

    return {
        "nose": _compute_for_level(nose_f),
        "ear":  _compute_for_level(ear_f),
    }


def build_basic_json(
    intraday: pd.DataFrame,
    ticker: str,
    mp_df: pd.DataFrame | None = None,
) -> dict:
    """
    Minimal JSON + MIDAS (anchor time, price, and F level)
    """

    # --- Base (same as original) ---
    if intraday is None or intraday.empty:
        total_vol = 0
        last_date = date.today()
    else:
        total_vol = int(intraday["Volume"].sum()) if "Volume" in intraday.columns else 0
        last_date = intraday["Date"].iloc[-1] if "Date" in intraday.columns else date.today()
        total_vol_readable = human_volume(total_vol)


    # ==========================
    # OPEN & CLOSE (real prices)
    # ==========================
    try:
        open_price = float(intraday["Open"].iloc[0])
    except:
        open_price = None

    try:
        close_price = float(intraday["Close"].iloc[-1])
    except:
        close_price = None

    # ==========================
    # MIDAS BEAR
    # ==========================
    try:
        bear_idx = intraday["MIDAS_Bear"].first_valid_index()
        if bear_idx is not None:
            midas_bear_time = intraday.loc[bear_idx, "Time"]
            midas_bear_f = float(intraday.loc[bear_idx, "F_numeric"])    # F level
            midas_bear_price = float(intraday.loc[bear_idx, "Close"])    # real price
        else:
            midas_bear_time = None
            midas_bear_f = None
            midas_bear_price = None
    except:
        midas_bear_time = midas_bear_f = midas_bear_price = None

    # ==========================
    # MIDAS BULL
    # ==========================
    try:
        bull_idx = intraday["MIDAS_Bull"].first_valid_index()
        if bull_idx is not None:
            midas_bull_time = intraday.loc[bull_idx, "Time"]
            midas_bull_f = float(intraday.loc[bull_idx, "F_numeric"])
            midas_bull_price = float(intraday.loc[bull_idx, "Close"])
        else:
            midas_bull_time = None
            midas_bull_f = None
            midas_bull_price = None
    except:
        midas_bull_time = midas_bull_f = midas_bull_price = None


    # ==========================
    # INITIAL BALANCE (IB)
    # ==========================
    try:
        # IB is always first 12 bars of intraday, matching your compute_initial_balance()
        ib_slice = intraday.iloc[:12]

        ib_high_f = float(ib_slice["F_numeric"].max())
        ib_low_f  = float(ib_slice["F_numeric"].min())

        # Locate time & real price for IB high
        ib_high_row = intraday.loc[intraday["F_numeric"] == ib_high_f].iloc[0]
        ib_high_time  = ib_high_row["Time"]
        ib_high_price = float(ib_high_row["Close"])

        # Locate time & real price for IB low
        ib_low_row  = intraday.loc[intraday["F_numeric"] == ib_low_f].iloc[0]
        ib_low_time   = ib_low_row["Time"]
        ib_low_price  = float(ib_low_row["Close"])

    except:
        ib_high_f = ib_low_f = None
        ib_high_time = ib_low_time = None
        ib_high_price = ib_low_price = None





    # ==========================
    # FINAL JSON
    # ==========================
    # ==========================
    # FINAL JSON
    # ==========================


    entries_block = extract_entries(intraday)
    milestones_block = extract_milestones(intraday)
    vector_cap_block = extract_vector_capacitance(intraday)
    mp_block = extract_market_profile(mp_df)
    profile_cross_block = extract_profile_cross_insight(intraday, mp_block)
    sector = detect_sector(ticker)
    slug = f"{ticker.lower()}-{last_date}-{sector}"
    expansion_block = detect_expansion_near_e1(intraday, perimeter=10)

    return round_all_numeric({
        "name": str(ticker).lower(),
        "date": str(last_date),
        "sector": sector,

        "slug": slug,

        "totalVolume": total_vol_readable,


        "open": open_price,
        "close": close_price,
        "expansionInsight": expansion_block,

        "entries": entries_block,
        "milestones": milestones_block,
        "milestones": milestones_block,
        "vectorCapacitance": vector_cap_block,
        "marketProfile": mp_block,
        "noseInsight": profile_cross_block.get("nose", {}),
        "earInsight":  profile_cross_block.get("ear",  {}),
        "initialBalance": {
            "high": {
                "time": ib_high_time,
                "fLevel": ib_high_f,
                "price": ib_high_price
            },
            "low": {
                "time": ib_low_time,
                "fLevel": ib_low_f,
                "price": ib_low_price
            }
        },

        "midas": {
            "bear": {
                "anchorTime": midas_bear_time,
                "price": midas_bear_price,
                "fLevel": midas_bear_f
            },
            "bull": {
                "anchorTime": midas_bull_time,
                "price": midas_bull_price,
                "fLevel": midas_bull_f
            }
        }
    })




def round_all_numeric(obj):
    if isinstance(obj, dict):
        return {k: round_all_numeric(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_all_numeric(x) for x in obj]
    else:
        try:
            return round(float(obj), 2)
        except:
            return obj

def render_json_batch_download(json_map: dict):
    """
    json_map = {
      "NVDA": {...},
      "SPY":  {...},
      ...
    }
    Renders one ZIP download button with all JSON files inside.
    """
    if not json_map:
        return

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for tkr, payload in json_map.items():
            # filename pattern: nvda-2025-11-21.json
            safe_name = payload.get("name", str(tkr)).lower()
            date_str = payload.get("date", "")
            if date_str:
                fname = f"{safe_name}-{date_str}.json"
            else:
                fname = f"{safe_name}.json"

            zf.writestr(fname, json.dumps(payload, indent=4, ensure_ascii=False)
)

    buffer.seek(0)

    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )
