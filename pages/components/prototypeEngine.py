# pages/components/prototypeEngine.py

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1) CORE DISPLACEMENT LOGIC (Ember / Cliff)
# ---------------------------------------------------------
def detect_core_prototype(intraday: pd.DataFrame) -> str:
    """
    Returns:
        "Ember"  → if F_numeric < -50 BEFORE first Call 🎯1
        "Cliff"  → if F_numeric > 50 BEFORE first Put 🎯1
        ""       → otherwise
    """

    if intraday is None or intraday.empty:
        return ""

    f = pd.to_numeric(intraday["F_numeric"], errors="coerce")

    call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    # No entries → no prototype
    if len(call_e1) == 0 and len(put_e1) == 0:
        return ""

    # Which entry happens first?
    all_e1 = []
    if len(call_e1) > 0:
        all_e1.append(("call", call_e1[0]))
    if len(put_e1) > 0:
        all_e1.append(("put", put_e1[0]))

    side, e1_idx = sorted(all_e1, key=lambda x: intraday.index.get_loc(x[1]))[0]
    e1_loc = intraday.index.get_loc(e1_idx)

    # Look BACKWARD (only before entry)
    before = f.iloc[:e1_loc]

    if side == "call":
        if (before < -50).any():
            return "Ember"
    if side == "put":
        if (before > 50).any():
            return "Cliff"

    return ""


# ---------------------------------------------------------
# 2) PREFIX: Tailbone
# ---------------------------------------------------------
def detect_tailbone(intraday: pd.DataFrame, profile_df, f_bins, pre_anchor_buffer=3) -> bool:
    """
    Tailbone = True if any Market Profile tail (🪶) exists in the F-bin range
    between (anchor - buffer) and Entry-1.
    """

    if intraday is None or intraday.empty:
        return False
    if profile_df is None or f_bins is None:
        return False

    f_vals = intraday["F_numeric"].to_numpy()

    # 1️⃣ Find first Entry-1 (call or put)
    call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    if len(call_e1) == 0 and len(put_e1) == 0:
        return False

    # earliest entry
    all_e1 = sorted(list(call_e1) + list(put_e1), key=lambda idx: intraday.index.get_loc(idx))
    e1_idx = all_e1[0]
    e1_loc = intraday.index.get_loc(e1_idx)

    # 2️⃣ Determine which anchor to use
    if e1_idx in call_e1:
        anchor_idx = intraday["MIDAS_Bull"].first_valid_index()
    else:
        anchor_idx = intraday["MIDAS_Bear"].first_valid_index()

    if anchor_idx is None:
        return False

    anchor_loc = intraday.index.get_loc(anchor_idx)

    # Entry must be after anchor
    if e1_loc <= anchor_loc - pre_anchor_buffer:
        return False

    # 3️⃣ Build window [anchor-3 … entry]
    lo = max(0, anchor_loc - pre_anchor_buffer)
    hi = e1_loc

    segment = f_vals[lo : hi + 1]

    # 4️⃣ Convert F-values into profile bins
    bin_ix = np.clip(np.digitize(segment, f_bins) - 1, 0, len(f_bins)-1)
    window_bins = pd.unique(f_bins[bin_ix])

    # 5️⃣ Extract bins that contain a Tail (🪶)
    tail_bins = set(profile_df.loc[profile_df["Tail"] == "🪶", "F% Level"].tolist())

    # 6️⃣ Tailbone = intersection
    return any(b in tail_bins for b in window_bins)

# ---------------------------------------------------------
# 3) PREFIX: Stampede
# ---------------------------------------------------------
def detect_stampede(intraday: pd.DataFrame, lookaround=7) -> bool:
    """
    Stampede = RVOL_5 spike > 1.2 within ± lookaround bars around first E1
    """

    call_e1 = intraday.index[intraday.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = intraday.index[intraday.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    if len(call_e1) == 0 and len(put_e1) == 0:
        return False

    # earliest entry
    all_e1 = sorted(list(call_e1) + list(put_e1), key=lambda idx: intraday.index.get_loc(idx))
    e1_idx = all_e1[0]
    e1_loc = intraday.index.get_loc(e1_idx)

    lo = max(0, e1_loc - lookaround)
    hi = min(len(intraday) - 1, e1_loc + lookaround)

    window = intraday.iloc[lo:hi+1]

    if "RVOL_5" not in window.columns:
        return False

    return (window["RVOL_5"] > 1.2).any()


# ---------------------------------------------------------
# 4) FINAL NAMING LOGIC
# ---------------------------------------------------------
def build_prototype_name(intraday: pd.DataFrame, profile_df=None, f_bins=None) -> str:
    """
    RULES:
    - Core = Ember / Cliff
    - Prefix = Tailbone OR Stampede OR ""
      (mutually exclusive; Tailbone wins if both true)
    """

    core = detect_core_prototype(intraday)
    if core == "":
        return ""   # no prototype

    # prefix logic
    is_tail = False
    is_stampede = False

    if profile_df is not None and f_bins is not None:
        is_tail = detect_tailbone(intraday, profile_df, f_bins)

    is_stampede = detect_stampede(intraday)

    if is_tail:
        return f"Tailbone {core}"

    if is_stampede:
        return f"Stampede {core}"

    return core
