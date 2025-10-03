import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io, re

# ---------- Fichiers ----------
FOR_FILE = Path("Labo3/Mesures/914b_donnees_iv_01.lvm")     # direct 0→1 V
REV_FILE = Path("Labo3/Mesures/914binv_donnees_iv_01.lvm")  # inverse 0→6 V
FIG_DIR  = Path("Labo3/Figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
RD_PNG   = FIG_DIR / "diode_resistance_dynamique.png"

# ---------- Lecture flexible .lvm ----------
def read_lvm_auto(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    num_re = re.compile(r"^\s*[-+]?(?:\d+|\d*[.,]\d+)\s")
    start = next(i for i, ln in enumerate(lines) if num_re.match(ln))
    block = "\n".join(lines[start:])
    for dec in [",", "."]:
        df = pd.read_csv(io.StringIO(block), sep=r"[\t;]+", engine="python", decimal=dec, header=None)
        num = df.select_dtypes("number")
        if num.shape[1] >= 2:
            num = num.iloc[:, :2].copy()
            a, b = num.iloc[:, 0].to_numpy(), num.iloc[:, 1].to_numpy()
            def mono_score(x):
                d = np.diff(x); return (d > 0).mean() if d.size else 0.0
            V, I = (b, a) if mono_score(b) > mono_score(a) else (a, b)
            return pd.DataFrame({"V": V.astype(float), "I": I.astype(float)}).dropna()
    return pd.DataFrame(columns=["V", "I"])

# ---------- Charge et oriente ----------
df_for = read_lvm_auto(FOR_FILE)
df_rev = read_lvm_auto(REV_FILE)

df_for["V"] =  np.abs(df_for["V"])
if df_for["I"].median() < 0:
    df_for["I"] = -df_for["I"]   # courant direct positif

df_rev["V"] = -np.abs(df_rev["V"])  # tensions négatives en inverse

# ---------- Combine, trie, consolide (pas ~1 mV) ----------
df = pd.concat([df_rev, df_for], ignore_index=True).sort_values("V").reset_index(drop=True)
df["Vg"] = df["V"].round(3)
df = df.groupby("Vg", as_index=False)["I"].mean().rename(columns={"Vg": "V"})

# ---------- Dérivée numérique et résistance dynamique ----------
SMOOTH_I  = 5   # lissage (points) sur I(V) avant dérivée
SMOOTH_RD = 5   # lissage (points) sur Rd(V) après dérivée
EPS = 1e-10

I_s = df["I"].rolling(window=SMOOTH_I, center=True, min_periods=1).median().bfill().ffill()
V   = df["V"].to_numpy()
I   = I_s.to_numpy()

if V.size >= 3:
    dIdV = np.gradient(I, V, edge_order=2)   # différences centrées
elif V.size == 2:
    dIdV = np.gradient(I, V, edge_order=1)
else:
    dIdV = np.full_like(I, np.nan)

Rd = np.where(np.isfinite(dIdV) & (np.abs(dIdV) > EPS), 1.0/np.abs(dIdV), np.nan)
Rd_s = pd.Series(Rd).rolling(window=SMOOTH_RD, center=True, min_periods=1).median().bfill().ffill().to_numpy()

# ---------- Tracé ----------
plt.figure(figsize=(9,6))
plt.plot(V, Rd_s, marker='o', markersize=3, linewidth=1)
plt.xlabel("Tension (v_d) [V]")
plt.ylabel("R_d = dV/dI (Ω)")
plt.title("Résistance dynamique de la diode standard")
plt.yscale("log")
plt.tight_layout()
plt.savefig(RD_PNG)
plt.show()
