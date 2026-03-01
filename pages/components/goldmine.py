import pandas as pd
import numpy as np

def apply_goldmine_e1(intraday: pd.DataFrame, dist: int = 64) -> pd.DataFrame:
    """
    💰 Goldmine Ladder from Entry 1 (🎯):
    Marks 💰 once every +dist F gain (call) or -dist F drop (put)
    AFTER the first Entry 1.

    Example:
        If dist=64:
        - First 💰 at anchor + 64
        - Next 💰 at anchor + 128
        - Next 💰 at anchor + 192
        etc.

        Same for puts but descending.
    """

    if intraday is None or intraday.empty:
        return intraday

    out = intraday.copy()
    out["Goldmine_E1_Emoji"] = ""

    # -------------------------
    # 1️⃣ Find first Entry-1
    # -------------------------
    call_e1 = out.index[out.get("Call_FirstEntry_Emoji", "") == "🎯"]
    put_e1  = out.index[out.get("Put_FirstEntry_Emoji", "")  == "🎯"]

    if len(call_e1) == 0 and len(put_e1) == 0:
        return out

    # Earliest of call or put
    all_e1 = sorted(list(call_e1) + list(put_e1), key=lambda idx: out.index.get_loc(idx))
    e1_idx = all_e1[0]

    is_call = e1_idx in call_e1
    is_put  = e1_idx in put_e1

    anchor = out.at[e1_idx, "F_numeric"]
    if pd.isna(anchor):
        return out

    # -------------------------
    # 2️⃣ Build ladder targets
    # -------------------------
    targets = []

    if is_call:
        # ascending: anchor+64, anchor+128, anchor+192, ...
        # build enough targets to cover entire day
        max_f = out["F_numeric"].max()
        t = anchor + dist
        while t <= max_f + dist:  # safety
            targets.append(t)
            t += dist

    elif is_put:
        # descending: anchor-64, anchor-128, anchor-192, ...
        min_f = out["F_numeric"].min()
        t = anchor - dist
        while t >= min_f - dist:
            targets.append(t)
            t -= dist

    # -------------------------
    # 3️⃣ Walk forward and mark each target once
    # -------------------------
    start_i = out.index.get_loc(e1_idx)
    last_hit = None
    target_i = 0  # step in the ladder

    for i in range(start_i + 1, len(out)):
        f = out.iloc[i]["F_numeric"]
        if pd.isna(f):
            continue

        # no more targets left
        if target_i >= len(targets):
            break

        target = targets[target_i]

        # Check hit
        if (is_call and f >= target) or (is_put and f <= target):
            out.iat[i, out.columns.get_loc("Goldmine_E1_Emoji")] = "💰"
            target_i += 1  # move to next target

    return out
