import pandas as pd
import numpy as np


# ============================================
# PROTOTYPE CONSTANTS
# ============================================

EMBER_DIP_MIN = -55
EMBER_DIP_MAX = -48
WINDOW = 5   # bars before entry


# ============================================
# VOLATILITY DETECTOR
# ============================================

def detect_volatility_flavor(intraday: pd.DataFrame, entry_index) -> str:
    """
    Returns one of:
      - "Ember"   (STD + BBW)
      - "Balloon" (BBW only)
      - "Stone"   (STD only)
      - ""        (none)
    """

    cols = ["STD_Alert", "BBW Alert"]
    for c in cols:
        if c not in intraday.columns:
            return ""

    entry_loc = intraday.index.get_loc(entry_index)
    start = max(0, entry_loc - WINDOW)
    window = intraday.iloc[start:entry_loc]

    std_present = window["STD_Alert"].astype(str).str.contains("🐦‍🔥").any()
    bbw_present = window["BBW Alert"].astype(str).str.contains("🔥").any()

    if std_present and bbw_present:
        return "Ember"

    if bbw_present and not std_present:
        return "Balloon"

    if std_present and not bbw_present:
        return "Stone"

    return ""


# ============================================
# DIP CHECK FOR EMBER ENVIRONMENT
# ============================================

def dipped_in_ember_zone(intraday: pd.DataFrame, entry_index) -> bool:
    """
    Checks if Mike dipped BETWEEN –55F and –48F prior to entry.
    """

    if "F_numeric" not in intraday.columns:
        return False

    entry_loc = intraday.index.get_loc(entry_index)
    history = intraday.iloc[:entry_loc]["F_numeric"]

    if history.empty:
        return False

    return ((history >= EMBER_DIP_MIN) & (history <= EMBER_DIP_MAX)).any()


# ============================================
# MAIN PROTOTYPE ASSIGNMENT
# ============================================

def assign_prototype_to_row(row, intraday: pd.DataFrame):
    """
    Outputs:
      - "Ember"
      - "" (no prototype)

    Also returns volatility type separately:
      - "Ember"
      - "Balloon"
      - "Stone"
      - ""
    """

    entry_type = row.get("Type", "")
    entry_time = row.get("Time", "")

    # Map entry to intraday index
    locs = intraday.index[
        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
    ]
    if len(locs) == 0:
        return "", ""

    entry_index = locs[0]

    # 1) Volatility flavor
    vol_type = detect_volatility_flavor(intraday, entry_index)

    # 2) Ember prototype check
    if entry_type == "Call 🎯1":
        if dipped_in_ember_zone(intraday, entry_index):
            if vol_type == "Ember":    # MUST be STD + BBW
                return "Ember", vol_type

    # default
    return "", vol_type
