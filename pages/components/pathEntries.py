# pages/components/pathEntries.py

import pandas as pd
import numpy as np


def apply_entry_paths(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all post-entry 'path' features directly to the INTRADAY dataframe.

    Returns intraday with new columns such as:
        - Door_Emoji
        - Door_Time
        - Door_Close

    Later we will add:
        - Bars_To_Door
        - Max_F_Before_Door
        - Max_Pain
        - UGD_Alert
        - etc.
    """
    if intraday is None or intraday.empty:
        return intraday

    # Ensure path columns exist with sane defaults
    if "Door_Emoji" not in intraday.columns:
        intraday["Door_Emoji"] = ""
    if "Door_Time" not in intraday.columns:
        intraday["Door_Time"] = ""
    if "Door_Close" not in intraday.columns:
        intraday["Door_Close"] = np.nan

    # If we don't have T0_Emoji at all, there are no doors to map
    if "T0_Emoji" not in intraday.columns:
        return intraday

    # --- Find all entry rows in intraday ---
    entry_mask = (
        (intraday.get("Put_FirstEntry_Emoji", "") == "🎯") |
        (intraday.get("Call_FirstEntry_Emoji", "") == "🎯") |
        (intraday.get("Put_SecondEntry_Emoji", "") == "🎯2") |
        (intraday.get("Call_SecondEntry_Emoji", "") == "🎯2") |
        (intraday.get("Put_ThirdEntry_Emoji", "") == "🎯3") |
        (intraday.get("Call_ThirdEntry_Emoji", "") == "🎯3")
    )

    entry_indices = intraday.index[entry_mask]

    # For each entry row, compute its path
    for idx in entry_indices:
        info = _map_stall_after_entry(idx, intraday)

        intraday.at[idx, "Door_Emoji"] = info["Door_Emoji"]
        intraday.at[idx, "Door_Time"] = info["Door_Time"]
        intraday.at[idx, "Door_Close"] = info["Door_Close"]

    return intraday


def _map_stall_after_entry(idx, intraday: pd.DataFrame) -> dict:
    """
    Given the index of the entry row, find the FIRST 🚪 after that row.
    """
    row_pos = intraday.index.get_loc(idx)

    # scan forward AFTER entry
    fwd = intraday.iloc[row_pos + 1 :]

    # Safety: T0_Emoji must exist here
    if "T0_Emoji" not in fwd.columns:
        return {"Door_Emoji": "", "Door_Time": "", "Door_Close": np.nan}

    hits = fwd[fwd["T0_Emoji"] == "🚪"]

    if hits.empty:
        return {"Door_Emoji": "", "Door_Time": "", "Door_Close": np.nan}

    r = hits.iloc[0]

    raw_time = r.get("Time", "")
    dt = pd.to_datetime(str(raw_time), errors="coerce")
    if pd.isna(dt):
        time_str = str(raw_time)
    else:
        time_str = dt.strftime("%H:%M")

    return {
        "Door_Emoji": "🚪",
        "Door_Time": time_str,
        "Door_Close": r.get("Close", np.nan),
    }
