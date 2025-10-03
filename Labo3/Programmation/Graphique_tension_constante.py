import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io, re, os

# ---------- Configuration ----------
DATA_FILE = Path("Labo3/Mesures/transistor_donnees_iv_01.lvm")
OUT_PNG = Path("Labo3/Figures/transistor_ic_vc_ipla.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ---------- Lecture LVM (entêtes NI, décimales ,/. ; tab/;) ----------
def read_lvm_flexible(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    start_idx = 0
    num_re = re.compile(r"^\s*[-+]?(\d+|\d*\.\d+|\d+,\d+)\s")
    for i, line in enumerate(lines):
        if num_re.match(line):
            start_idx = i
            break
    data_str = "\n".join(lines[start_idx:])
    for dec in [",", "."]:
        try:
            df = pd.read_csv(io.StringIO(data_str), decimal=dec, sep=r"\t|;", engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    for dec in [",", "."]:
        try:
            df = pd.read_csv(path, decimal=dec, sep=r"\t|;", engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    raise RuntimeError("Impossible de lire automatiquement le .lvm")

def normalize_transistor_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.dropna(how="all", axis=1)
    if df2.shape[1] > 3:
        na_counts = df2.isna().sum().sort_values()
        df2 = df2[na_counts.index[:3]]
    if df2.shape[1] == 3:
        df2.columns = ["Vc", "Ic", "Vb"]
    elif df2.shape[1] == 2:
        df2.columns = ["Vc", "Ic"]
    else:
        df2.columns = [f"col{i}" for i in range(1, df2.shape[1]+1)]
    return df2

def group_by_vb(df: pd.DataFrame, vb_round=2):
    if "Vb" in df.columns:
        vb_vals = np.round(df["Vb"].astype(float), vb_round)
        groups = {}
        for vb in np.unique(vb_vals[~np.isnan(vb_vals)]):
            sub = df.loc[vb_vals==vb, ["Vc","Ic"]].dropna().sort_values("Vc")
            if not sub.empty:
                groups[float(vb)] = sub
        return groups
    # Sinon : blocs consécutifs où Vc « redémarre »
    vc = df["Vc"].to_numpy()
    breaks = [0]
    for i in range(1, len(vc)):
        if vc[i] < 0.2*vc[i-1]:
            breaks.append(i)
    breaks.append(len(vc))
    groups = {}
    for k in range(len(breaks)-1):
        sub = df.iloc[breaks[k]:breaks[k+1]][["Vc","Ic"]].dropna().sort_values("Vc")
        if not sub.empty:
            vb_label = float(np.round(sub["Ic"].head(max(2, int(0.1*len(sub)))).mean(), 3))
            groups[vb_label] = sub
    return groups

def plateau_estimate_ic(sub_df: pd.DataFrame, frac=0.2):
    n = len(sub_df)
    if n < 5:
        return float(sub_df["Ic"].mean()), 0.0
    k = max(3, int(np.ceil(frac*n)))
    tail = sub_df.sort_values("Vc").tail(k)
    ipla = float(tail["Ic"].mean())
    sigma = float(tail["Ic"].std(ddof=1)) if k > 2 else 0.0
    return ipla, sigma

# ---------- Chargement ----------
raw = read_lvm_flexible(DATA_FILE)
df = normalize_transistor_columns(raw)
groups = group_by_vb(df, vb_round=2)

# ---------- Tracé ----------
plt.figure(figsize=(10,6))
annotations = []
summary = []

for vb, sub in sorted(groups.items(), key=lambda kv: kv[0]):
    x = sub["Vc"].to_numpy()
    y = sub["Ic"].to_numpy()
    plt.plot(x, y, linestyle='-', label=f"Vb = {vb:.2f} V")
    ipla, sigma = plateau_estimate_ic(sub, frac=0.2)
    annotations.append((float(x.max()), ipla))
    summary.append({"Vb (V)": vb, "i_pla (A)": ipla, "sigma (A)": sigma, "N_points": len(sub)})

for xr, ipla in annotations:
    plt.annotate(f"iₚₗₐ ≈ {ipla:.3g} A", xy=(xr, ipla), xytext=(xr, ipla),
                 textcoords="data", ha="left", va="bottom")

plt.xlabel("v_C (V)")
plt.ylabel("i_C (A)")
plt.title("Transistor 2219A — Courbes i_C–v_C (par V_B)\nPlateau iₚₗₐ = moyenne du dernier 20 % en v_C")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG)