import pandas as pd
import numpy as np

def apply_parallel_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks ⚡ Parallel phase:
    After T1, as long as price stays on the correct side of Tenkan_F.
      - For calls: F_numeric >= Tenkan_F
      - For puts:  F_numeric <= Tenkan_F
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["Parallel_Emoji"] = ""

    # Find T1
    if "T1_Emoji" not in out.columns:
        return out
    t1_idx = out.index[out["T1_Emoji"] == "🏇🏼"]
    if len(t1_idx) == 0:
        return out

    start_i = out.index.get_loc(t1_idx[0])

    # Determine side from Entry 1
    side = None
    if "Call_FirstEntry_Emoji" in out.columns and any(out["Call_FirstEntry_Emoji"] == "🎯"):
        side = "call"
    elif "Put_FirstEntry_Emoji" in out.columns and any(out["Put_FirstEntry_Emoji"] == "🎯"):
        side = "put"

    if side is None:
        return out

      # Need F_numeric
    if "F_numeric" not in out.columns:
        return out

    # Detect Tenkan column name
    if "Tenkan_F" in out.columns:
        tenkan_col = "Tenkan_F"
    elif "F% Tenkan" in out.columns:
        tenkan_col = "F% Tenkan"
    else:
        # no Tenkan available → no parallel logic
        return out

    # Loop forward bar by bar after T1
    for i in range(start_i + 1, len(out)):
        idx = out.index[i]
        mike = out.at[idx, "F_numeric"]
        tenkan = out.at[idx, tenkan_col]

        if pd.isna(mike) or pd.isna(tenkan):
            continue

        if side == "call" and mike >= tenkan:
            out.at[idx, "Parallel_Emoji"] = "⚡"
        elif side == "put" and mike <= tenkan:
            out.at[idx, "Parallel_Emoji"] = "⚡"
        else:
            # exits parallel phase
            break


        if pd.isna(mike) or pd.isna(tenkan):
            continue

        if side == "call" and mike >= tenkan:
            out.at[idx, "Parallel_Emoji"] = "⚡"
        elif side == "put" and mike <= tenkan:
            out.at[idx, "Parallel_Emoji"] = "⚡"
        else:
            # exits parallel phase
            break

    return out
