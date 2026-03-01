# pages/components/jsonExport.py

import io
import json
import zipfile
from datetime import date

import pandas as pd
import streamlit as st

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

def build_basic_json(intraday: pd.DataFrame, ticker: str) -> dict:
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

        entries_block = extract_entries(intraday)




    # ==========================
    # FINAL JSON
    # ==========================
    # ==========================
    # FINAL JSON
    # ==========================
    return round_all_numeric({
        "name": str(ticker).lower(),
        "date": str(last_date),
        "totalVolume": total_vol_readable,


        "open": open_price,
        "close": close_price,
        "entries": entries_block,    
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

            zf.writestr(fname, json.dumps(payload, indent=4))

    buffer.seek(0)

    st.download_button(
        label="⬇️ Download JSON batch",
        data=buffer,
        file_name="mike_json_batch.zip",
        mime="application/zip",
    )
