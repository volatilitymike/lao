import pandas as pd
import numpy as np
import string


def compute_market_profile(
    intraday: pd.DataFrame,
    price_col: str = "F_numeric",
) -> tuple[pd.DataFrame, dict]:
    """
    Market Profile (TPO) component:

    - Uses `price_col` (default F_numeric) as Mike axis.
    - F% bins: 20-point bins from -400 to +400.
    - 15-min TPO letters: A, B, C… (AA if > 26 brackets).
    - Initial Balance (IB): letters A–D.
    - IB thirds (Upper / Middle / Lower).
    - TPO letters per F% level.
    - Tails (🪶) when there's ONLY one TPO at that level.
    - TPO_Count = number of letters at that level.
    - %Vol = volume per F% bin (normalized).
    - ✅ ValueArea = 70% of total TPOs by count, built from POC outward.
    - 🦻🏼 Ear = dominant volume bin (VPOC) only if Mike moved away.
    - 👃🏽 Nose = dominant time bin (POC) only if Mike moved away.

    NEW:
    - 💥 Range_Extension: post-IB TPOs outside IB range.
    - IB_High_Break / IB_Low_Break flags.
    - Stamina: % of TPOs that happened AFTER IB.
    - Stamina_Signal:
        - 💪 strong post-IB conviction
        - 🥀 faded after IB

    Returns:
        profile_df: one row per F% level.
        ib_info: IB stats + POC / VA info.
    """

    df = intraday.copy()

    # ---------------------------------
    # Guard rails
    # ---------------------------------
    if df.empty or "Time" not in df.columns or price_col not in df.columns:
        cols = [
            "F% Level", "Letters", "Tail", "TPO_Count", "%Vol",
            "✅ ValueArea", "🦻🏼", "👃🏽",
            "Range_Extension", "IB_High_Break", "IB_Low_Break",
            "Stamina", "Stamina_Signal",
        ]
        empty = pd.DataFrame(columns=cols)
        ib_info = {
            "IB_High": np.nan,
            "IB_Low": np.nan,
            "IB_Mid_Third": np.nan,
            "IB_Top_Third": np.nan,
            "POC_F_Level": np.nan,
            "VPOC_F_Level": np.nan,
            "VA_Low_F": np.nan,
            "VA_High_F": np.nan,
        }
        return empty, ib_info

    df = df[df[price_col].notna()].copy()
    if df.empty:
        return compute_market_profile(df, price_col)  # will hit guard above

    # ---------------------------------
    # 1) TimeIndex & LetterIndex
    # ---------------------------------
    df["TimeIndex"] = pd.to_datetime(df["Time"], format="%I:%M %p", errors="coerce")
    df = df[df["TimeIndex"].notna()].copy()
    if df.empty:
        return compute_market_profile(df, price_col)

    # 15-min brackets
    df["LetterIndex"] = (
        (df["TimeIndex"].dt.hour * 60 + df["TimeIndex"].dt.minute) // 15
    ).astype(int)
    df["LetterIndex"] -= df["LetterIndex"].min()

    def letter_code(n: int) -> str:
        letters = string.ascii_uppercase
        if n < 26:
            return letters[n]
        return letters[(n // 26) - 1] + letters[n % 26]

    df["Letter"] = df["LetterIndex"].apply(letter_code)

    # ---------------------------------
    # 2) Initial Balance (A–D)
    # ---------------------------------
    ib_letters = ["A", "B", "C", "D"]
    ib_df = df[df["Letter"].isin(ib_letters)]

    if not ib_df.empty:
        IB_High = float(ib_df[price_col].max())
        IB_Low = float(ib_df[price_col].min())
    else:
        IB_High = np.nan
        IB_Low = np.nan

    if pd.notna(IB_High) and pd.notna(IB_Low) and IB_High != IB_Low:
        one_third = (IB_High - IB_Low) / 3
        IB_Mid_Third = IB_Low + one_third
        IB_Top_Third = IB_Low + 2 * one_third
    else:
        IB_Mid_Third = np.nan
        IB_Top_Third = np.nan

    ib_info = {
        "IB_High": IB_High,
        "IB_Low": IB_Low,
        "IB_Mid_Third": IB_Mid_Third,
        "IB_Top_Third": IB_Top_Third,
    }

    # ---------------------------------
    # 3) F% bins & letters per bin
    # ---------------------------------
    f_bins = np.arange(-400, 401, 20)  # -400, -380, ..., +380
    df["F_Bin"] = pd.cut(
        df[price_col],
        bins=f_bins,
        labels=[str(x) for x in f_bins[:-1]],
    )

    profile = {}
    for level in f_bins[:-1]:
        s = str(level)
        letters = (
            df.loc[df["F_Bin"] == s, "Letter"]
              .dropna()
              .unique()
        )
        if len(letters) > 0:
            profile[s] = "".join(sorted(letters))

    profile_df = pd.DataFrame(
        list(profile.items()),
        columns=["F% Level", "Letters"],
    )

    if profile_df.empty:
        profile_df["F% Level"] = []
        for col in [
            "Letters", "Tail", "TPO_Count", "%Vol",
            "✅ ValueArea", "🦻🏼", "👃🏽",
            "Range_Extension", "IB_High_Break", "IB_Low_Break",
            "Stamina", "Stamina_Signal",
        ]:
            profile_df[col] = []
        return profile_df, ib_info

    profile_df["F% Level"] = profile_df["F% Level"].astype(int)

    # ---------------------------------
    # 4) Tails & TPO count
    # ---------------------------------
    def tail_marker(letters: str) -> str:
        # Real tail: exactly ONE TPO (one letter)
        if not isinstance(letters, str):
            return ""
        return "🪶" if len(letters) == 1 else ""

    profile_df["Tail"] = profile_df["Letters"].apply(tail_marker)
    profile_df["TPO_Count"] = profile_df["Letters"].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )

    # ---------------------------------
    # 5) %Vol per bin
    # ---------------------------------
    if "Volume" in df.columns:
        vol_per_bin = df.groupby("F_Bin")["Volume"].sum()
        total_vol = vol_per_bin.sum() if vol_per_bin.sum() != 0 else 1.0
        vol_percent = (vol_per_bin / total_vol * 100).round(2)
        profile_df["%Vol"] = (
            profile_df["F% Level"].astype(str)
            .map(vol_percent)
            .fillna(0)
        )
    else:
        profile_df["%Vol"] = 0.0

    # ---------------------------------
    # 6) POC, Value Area 70% from POC outward
    # ---------------------------------
    profile_df = profile_df.sort_values("F% Level").reset_index(drop=True)

    tpo = profile_df["TPO_Count"].to_numpy()
    levels = profile_df["F% Level"].to_numpy()
    total_tpos = tpo.sum()
    target_tpos = total_tpos * 0.70 if total_tpos > 0 else 0

    # TPO POC (time POC)
    if total_tpos > 0:
        poc_idx = int(tpo.argmax())
        poc_level = int(levels[poc_idx])
    else:
        poc_idx = 0
        poc_level = int(levels[0])

    # Build VA from POC outward
    n = len(tpo)
    va_indices = set()
    cum = 0

    if total_tpos > 0:
        va_indices.add(poc_idx)
        cum = tpo[poc_idx]
        step = 1
        while cum < target_tpos and (poc_idx - step >= 0 or poc_idx + step < n):
            candidates = []
            below = poc_idx - step
            above = poc_idx + step
            if below >= 0:
                candidates.append(below)
            if above < n:
                candidates.append(above)
            if not candidates:
                break
            # choose side with more TPOs, tie → closer to POC (same distance anyway)
            best = max(candidates, key=lambda i: tpo[i])
            if best in va_indices:  # just in case
                step += 1
                continue
            va_indices.add(best)
            cum += tpo[best]
            step += 1

    va_mask = np.array([i in va_indices for i in range(n)]) if va_indices else np.zeros(n, bool)
    profile_df["✅ ValueArea"] = np.where(va_mask, "✅", "")

    if va_mask.any():
        VA_Low_F = int(levels[va_mask].min())
        VA_High_F = int(levels[va_mask].max())
    else:
        VA_Low_F = VA_High_F = np.nan

    # Volume POC (VPOC)
    if profile_df["%Vol"].sum() > 0:
        v_idx = int(profile_df["%Vol"].to_numpy().argmax())
        vpoc_level = int(profile_df.loc[v_idx, "F% Level"])
    else:
        vpoc_level = poc_level

    # ---------------------------------
    # 7) Range Extension & IB breakouts
    # ---------------------------------
    def ib_split_counts(letters: str) -> tuple[int, int]:
        if not isinstance(letters, str):
            return 0, 0
        ib_set = set("ABCD")
        ib = sum(1 for ch in letters if ch in ib_set)
        post = len(letters) - ib
        return ib, post

    ib_counts, post_counts = zip(*profile_df["Letters"].apply(ib_split_counts))
    ib_counts = np.array(ib_counts)
    post_counts = np.array(post_counts)

    profile_df["IB_TPO"] = ib_counts
    profile_df["PostIB_TPO"] = post_counts

    def row_range_ext(level, post_tpo) -> str:
        if pd.isna(IB_High) or pd.isna(IB_Low):
            return ""
        # post-IB activity outside IB range
        if post_tpo > 0 and (level > IB_High or level < IB_Low):
            return "💥"
        return ""

    profile_df["Range_Extension"] = [
        row_range_ext(lvl, post)
        for lvl, post in zip(profile_df["F% Level"], profile_df["PostIB_TPO"])
    ]

    def ib_break_high(level, post_tpo) -> str:
        if pd.isna(IB_High) or post_tpo <= 0:
            return ""
        return "🔺" if level > IB_High else ""

    def ib_break_low(level, post_tpo) -> str:
        if pd.isna(IB_Low) or post_tpo <= 0:
            return ""
        return "🔻" if level < IB_Low else ""

    profile_df["IB_High_Break"] = [
        ib_break_high(lvl, post)
        for lvl, post in zip(profile_df["F% Level"], profile_df["PostIB_TPO"])
    ]
    profile_df["IB_Low_Break"] = [
        ib_break_low(lvl, post)
        for lvl, post in zip(profile_df["F% Level"], profile_df["PostIB_TPO"])
    ]

    # ---------------------------------
    # 8) Stamina (post-IB conviction vs fade)
    # ---------------------------------
    def stamina_value(total_tpo, post_tpo) -> float:
        if total_tpo <= 0:
            return 0.0
        return round(100.0 * post_tpo / total_tpo, 1)

    profile_df["Stamina"] = [
        stamina_value(t, post)
        for t, post in zip(profile_df["TPO_Count"], profile_df["PostIB_TPO"])
    ]

    def stamina_signal(total_tpo, ib_tpo, post_tpo, stamina) -> str:
        # full fade: action only in IB, nothing after, but with some structure
        if total_tpo >= 2 and post_tpo == 0 and ib_tpo > 0:
            return "🥀"
        # strong stamina: mostly post-IB activity
        if stamina >= 70 and post_tpo >= 2:
            return "💪"
        return ""

    profile_df["Stamina_Signal"] = [
        stamina_signal(t, ib, post, st)
        for t, ib, post, st in zip(
            profile_df["TPO_Count"],
            profile_df["IB_TPO"],
            profile_df["PostIB_TPO"],
            profile_df["Stamina"],
        )
    ]

    # ---------------------------------
    # 9) Ear (🦻🏼) & Nose (👃🏽) – only if Mike moved away
    # ---------------------------------
    current_mike = float(df[price_col].iloc[-1])
    EAR_NOSE_THRESHOLD = 40.0  # Mike distance in F% points

    profile_df["🦻🏼"] = ""
    profile_df["👃🏽"] = ""

    def ear_marker(level: int) -> str:
        if abs(current_mike - level) >= EAR_NOSE_THRESHOLD and level == vpoc_level:
            return "🦻🏼"
        return ""

    def nose_marker(level: int) -> str:
        if abs(current_mike - level) >= EAR_NOSE_THRESHOLD and level == poc_level:
            return "👃🏽"
        return ""

    profile_df["🦻🏼"] = profile_df["F% Level"].apply(ear_marker)
    profile_df["👃🏽"] = profile_df["F% Level"].apply(nose_marker)

    # ---------------------------------
    # 10) Final sort & ib_info enrich
    # ---------------------------------
    profile_df = profile_df.sort_values("F% Level").reset_index(drop=True)

    ib_info.update(
        {
            "POC_F_Level": poc_level,
            "VPOC_F_Level": vpoc_level,
            "VA_Low_F": VA_Low_F,
            "VA_High_F": VA_High_F,
        }
    )

    return profile_df, ib_info
