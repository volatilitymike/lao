# pages/components/entrySystem.py

import numpy as np
import pandas as pd


def apply_entry_system(
    intraday: pd.DataFrame,
    ib_info: dict | None = None,
    use_physics: bool = True,
) -> pd.DataFrame:
    """
    Attach Mike's Entry 1 / Entry 2 / Entry 3 markers:

      🎯   = FirstEntry (after MIDAS anchor + Heaven/Drizzle)
      🎯2  = SecondEntry (Kijun cross continuation after 🎯)
      🎯3  = ThirdEntry (IB break after 🎯 + 🎯2)

    Optional physics filters on 🎯2 / 🎯3 use:
      - Vector_pct (direction)
      - Volatility_Composite (for E3 regime check)
    """

    if intraday is None or intraday.empty:
        return intraday

    df = intraday.copy()

    # ---------------------------------
    # 0) Ensure core numeric columns
    # ---------------------------------
    df["F_numeric"] = pd.to_numeric(df.get("F_numeric"), errors="coerce")
    if "Kijun_F" in df.columns:
        df["Kijun_F"] = pd.to_numeric(df["Kijun_F"], errors="coerce")

    # ---------------------------------
    # 1) Attach IB_High / IB_Low columns if we got ib_info
    # ---------------------------------
    ib_high = ib_low = None
    if ib_info is not None:
        # support both styles: IB_High / ib_high
        ib_high = ib_info.get("IB_High", ib_info.get("ib_high"))
        ib_low = ib_info.get("IB_Low", ib_info.get("ib_low"))

    if "IB_High" not in df.columns:
        df["IB_High"] = np.nan
    if "IB_Low" not in df.columns:
        df["IB_Low"] = np.nan

    if ib_high is not None:
        df["IB_High"] = float(ib_high)
    if ib_low is not None:
        df["IB_Low"] = float(ib_low)

    # ---------------------------------
    # 2) Prepare entry emoji columns
    # ---------------------------------
    for col in [
        "Put_FirstEntry_Emoji",
        "Call_FirstEntry_Emoji",
        "Put_SecondEntry_Emoji",
        "Call_SecondEntry_Emoji",
        "Put_ThirdEntry_Emoji",
        "Call_ThirdEntry_Emoji",
    ]:
        if col not in df.columns:
            df[col] = ""

    # ---------------------------------
    # 3) Small helpers for physics filters
    # ---------------------------------
    def last_vector_sign(idx: int) -> int | None:
        """
        Look backwards up to idx and find the last non-null Vector_pct.
        Returns:
            +1 → last vector was upward
            -1 → last vector was downward
             0 → flat
            None → no vector info / physics disabled
        """
        if not use_physics:
            return None
        if "Vector_pct" not in df.columns:
            return None

        # idx is positional here (iloc index)
        series = df["Vector_pct"]
        if series.isna().all():
            return None

        # up to this bar
        sub = series.iloc[: idx + 1].dropna()
        if sub.empty:
            return None

        val = sub.iloc[-1]
        if val > 0:
            return 1
        if val < 0:
            return -1
        return 0

    def vol_composite_ok(idx: int) -> bool:
        """
        For Entry 3: require Volatility_Composite to be at least
        around its rolling median, so we only trade IB breaks in
        an active regime, not in dead tape.
        """
        if not use_physics:
            return True

        col = "Volatility_Composite"
        if col not in df.columns:
            return True

        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            return True

        window = 20
        med = series.rolling(window, min_periods=max(5, window // 2)).median()
        if np.isnan(med.iloc[idx]):
            return True

        return bool(series.iloc[idx] >= med.iloc[idx])

    # ---------------------------------
    # 4) ENTRY 1 – MIDAS anchor + Heaven/Drizzle
    # ---------------------------------
    # Puts: MIDAS_Bear → first Drizzle_Emoji 🌧️
    if "MIDAS_Bear" in df.columns and "Drizzle_Emoji" in df.columns:
        anchor_idx = df["MIDAS_Bear"].first_valid_index()
        if anchor_idx is not None:
            start_pos = df.index.get_loc(anchor_idx)
            for i in range(start_pos, len(df)):
                if df.iloc[i]["Drizzle_Emoji"] == "🌧️":
                    df.at[df.index[i], "Put_FirstEntry_Emoji"] = "🎯"
                    break

    # Calls: MIDAS_Bull → first Heaven_Cloud ☁️
    if "MIDAS_Bull" in df.columns and "Heaven_Cloud" in df.columns:
        anchor_idx = df["MIDAS_Bull"].first_valid_index()
        if anchor_idx is not None:
            start_pos = df.index.get_loc(anchor_idx)
            for i in range(start_pos, len(df)):
                if df.iloc[i]["Heaven_Cloud"] == "☁️":
                    df.at[df.index[i], "Call_FirstEntry_Emoji"] = "🎯"
                    break

    # ---------------------------------
    # 5) ENTRY 2 – SIMPLE KIJUN CROSS AFTER ENTRY 1 (NO PHYSICS)
    # ---------------------------------
    f = df["F_numeric"]
    kijun = df.get("Kijun_F")

    # --- PUT 🎯2 ---
    first_put = df.index[df["Put_FirstEntry_Emoji"] == "🎯"]
    if len(first_put) > 0 and kijun is not None:
        start_loc = df.index.get_loc(first_put[0])

        for i in range(start_loc + 1, len(df)):
            prev_f = f.iloc[i - 1]
            curr_f = f.iloc[i]
            prev_k = kijun.iloc[i - 1]
            curr_k = kijun.iloc[i]

            if (
                pd.notna(prev_f)
                and pd.notna(curr_f)
                and pd.notna(prev_k)
                and pd.notna(curr_k)
            ):
                # First cross *down* Kijun_F after E1
                if prev_f > prev_k and curr_f <= curr_k:
                    df.at[df.index[i], "Put_SecondEntry_Emoji"] = "🎯2"
                    break

    # --- CALL 🎯2 ---
    first_call = df.index[df["Call_FirstEntry_Emoji"] == "🎯"]
    if len(first_call) > 0 and kijun is not None:
        start_loc = df.index.get_loc(first_call[0])

        for i in range(start_loc + 1, len(df)):
            prev_f = f.iloc[i - 1]
            curr_f = f.iloc[i]
            prev_k = kijun.iloc[i - 1]
            curr_k = kijun.iloc[i]

            if (
                pd.notna(prev_f)
                and pd.notna(curr_f)
                and pd.notna(prev_k)
                and pd.notna(curr_k)
            ):
                # First cross *up* Kijun_F after E1
                if prev_f < prev_k and curr_f >= curr_k:
                    df.at[df.index[i], "Call_SecondEntry_Emoji"] = "🎯2"
                    break

    # ---------------------------------
    # 6) ENTRY 3 – IB break after 🎯 + 🎯2
    #     with physics direction + Volatility_Composite regime
    # ---------------------------------
    ib_low_series = pd.to_numeric(df["IB_Low"], errors="coerce")
    ib_high_series = pd.to_numeric(df["IB_High"], errors="coerce")

    # --- PUT 🎯3 (IB_Low break) ---
    first_put = df.index[df["Put_FirstEntry_Emoji"] == "🎯"]
    second_put = df.index[df["Put_SecondEntry_Emoji"] == "🎯2"]

    if len(first_put) > 0 and len(second_put) > 0:
        i_first = df.index.get_loc(first_put[0])
        i_second = df.index.get_loc(second_put[0])

        # Check if IB_Low already crossed between E1 and E2
        ib_low_crossed_by_second = False
        for i in range(i_first, i_second + 1):
            val_f = f.iloc[i]
            val_ib = ib_low_series.iloc[i]
            if pd.notna(val_f) and pd.notna(val_ib) and val_f < val_ib:
                ib_low_crossed_by_second = True
                break

        if not ib_low_crossed_by_second:
            for i in range(i_second + 1, len(df) - 1):
                f_prev = f.iloc[i - 1]
                f_curr = f.iloc[i]
                ib_prev = ib_low_series.iloc[i - 1]
                ib_curr = ib_low_series.iloc[i]

                if (
                    pd.notna(f_prev)
                    and pd.notna(f_curr)
                    and pd.notna(ib_prev)
                    and pd.notna(ib_curr)
                ):
                    # First cross below IB_Low
                    if f_prev > ib_prev and f_curr <= ib_curr:
                        j = i + 1
                        f_next = f.iloc[j]
                        if pd.notna(f_next) and f_next < f_curr:
                            ok = True
                            if use_physics:
                                sign = last_vector_sign(j)
                                if sign is not None and sign >= 0:
                                    ok = False
                                if ok and not vol_composite_ok(j):
                                    ok = False
                            if ok:
                                df.at[df.index[j], "Put_ThirdEntry_Emoji"] = "🎯3"
                                break

    # --- CALL 🎯3 (IB_High break) ---
    first_call = df.index[df["Call_FirstEntry_Emoji"] == "🎯"]
    second_call = df.index[df["Call_SecondEntry_Emoji"] == "🎯2"]

    if len(first_call) > 0 and len(second_call) > 0:
        i_first = df.index.get_loc(first_call[0])
        i_second = df.index.get_loc(second_call[0])

        # Check if IB_High already crossed between E1 and E2
        crossed_by_second = False
        for i in range(i_first, i_second + 1):
            val_f = f.iloc[i]
            val_ib = ib_high_series.iloc[i]
            if pd.notna(val_f) and pd.notna(val_ib) and val_f > val_ib:
                crossed_by_second = True
                break

        if not crossed_by_second:
            for i in range(i_second + 1, len(df) - 1):
                f_prev = f.iloc[i - 1]
                f_curr = f.iloc[i]
                ib_prev = ib_high_series.iloc[i - 1]
                ib_curr = ib_high_series.iloc[i]

                if (
                    pd.notna(f_prev)
                    and pd.notna(f_curr)
                    and pd.notna(ib_prev)
                    and pd.notna(ib_curr)
                ):
                    # First cross above IB_High
                    if f_prev < ib_prev and f_curr >= ib_curr:
                        j = i + 1
                        f_next = f.iloc[j]
                        if pd.notna(f_next) and f_next > f_curr:
                            ok = True
                            if use_physics:
                                sign = last_vector_sign(j)
                                if sign is not None and sign <= 0:
                                    ok = False
                                if ok and not vol_composite_ok(j):
                                    ok = False
                            if ok:
                                df.at[df.index[j], "Call_ThirdEntry_Emoji"] = "🎯3"
                                break

    return df
