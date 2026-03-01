import pandas as pd
import numpy as np


def map_parallel_after_t2(row, intraday: pd.DataFrame):
    """
    For a given entry row, find T2 (⚡), then track the Parallel phase.
    Returns (emoji, end_time, max_gain_F).

    - Find this entry bar by Time (HH:MM).
    - From there, find the first T2 (⚡).
    - From that T2 forward, look for Parallel_Emoji == ⚡.
    - If none: return ⚡, T2 time, 0 gain.
    - If some: return ⚡, last parallel time, max(F_numeric) - F_at_T2.
    """
    entry_time = row["Time"]

    # locate the entry bar by HH:MM
    locs = intraday.index[
        pd.to_datetime(intraday["Time"]).dt.strftime("%H:%M") == entry_time
    ]
    if len(locs) == 0:
        return pd.Series(["", "", ""])

    entry_idx = locs[0]
    entry_loc = intraday.index.get_loc(entry_idx)

    # scan forward for the first T2 ⚡how to pass it to d
    fwd = intraday.iloc[entry_loc + 1 :]
    hits_t2 = fwd[fwd.get("T2_Emoji", "") == "⚡"]
    if hits_t2.empty:
        return pd.Series(["", "", ""])

    # take first T2
    t2 = hits_t2.iloc[0]
    t2_loc = intraday.index.get_loc(t2.name)

    # from T2 onward, track Parallel phase
    fwd2 = intraday.iloc[t2_loc + 1 :]
    parallels = fwd2[fwd2.get("Parallel_Emoji", "") == "⚡"]
    if parallels.empty:
        # no parallel bars; end at T2, gain = 0
        return pd.Series([
            "⚡",
            pd.to_datetime(t2["Time"]).strftime("%H:%M"),
            0,
        ])

    # compute max gain relative to T2
    base_f = t2["F_numeric"]
    max_f = parallels["F_numeric"].max() if "F_numeric" in parallels else base_f
    gain_f = max_f - base_f

    last = parallels.iloc[-1]
    return pd.Series([
        "⚡",
        pd.to_datetime(last["Time"]).strftime("%H:%M"),
        gain_f,
    ])
