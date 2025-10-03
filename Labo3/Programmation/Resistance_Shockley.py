import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, re
import os

from pathlib import Path


# --- Fichiers / sortie ---
FOR_FILE = Path("Labo3/Mesures/914b_donnees_iv_01.lvm")     # direct 0→1 V
REV_FILE = Path("Labo3/Mesures/914binv_donnees_iv_01.lvm")  # inverse 0→6 V
FIG_DIR  = Path("Labo3/Figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = FIG_DIR / "diode_iv_shockley_fit.png"

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
            a, b = num.iloc[:,0].to_numpy(), num.iloc[:,1].to_numpy()
            def mono_score(x):
                d = np.diff(x); return (d>0).mean() if d.size else 0.0
            V, I = (b, a) if mono_score(b) > mono_score(a) else (a, b)
            return pd.DataFrame({"V": V.astype(float), "I": I.astype(float)}).dropna()
    return pd.DataFrame(columns=["V","I"])

# charge / oriente / combine
df_for = read_lvm_auto(FOR_FILE)
df_rev = read_lvm_auto(REV_FILE)

df_for["V"] =  np.abs(df_for["V"])
if df_for["I"].median() < 0:
    df_for["I"] = -df_for["I"]
df_rev["V"] = -np.abs(df_rev["V"])

df = pd.concat([df_rev, df_for], ignore_index=True).sort_values("V").reset_index(drop=True)

# Fit Shockley i(v)=i0 (e^{v/v0}-1) sur la branche directe
fwd = df[(df["V"] > 0) & (df["I"] > 0)].copy()
if len(fwd) < 3:
    raise RuntimeError("Pas assez de points en direct pour l'ajustement.")

v20, v85 = float(fwd["V"].quantile(0.20)), float(fwd["V"].quantile(0.85))
fit = fwd[(fwd["V"] >= max(0.20, v20)) & (fwd["V"] <= v85)].copy()
if len(fit) < 3:
    fit = fwd

lnI = np.log(fit["I"].to_numpy())
Vx  = fit["V"].to_numpy()
slope, intercept = np.polyfit(Vx, lnI, 1)    # ln(I) ~ ln(i0) + V/v0
v0 = 1.0 / slope
i0 = float(np.exp(intercept))

V_plot = np.linspace(df["V"].min(), df["V"].max(), 600)
I_fit  = np.where(V_plot >= 0, i0*(np.exp(V_plot/v0) - 1.0), np.nan)

# Tracé I–V + fit
plt.figure(figsize=(9,6))
plt.plot(df["V"], df["I"], 'o', ms=3, lw=1, label="Courbe i-v de la diode")
# ⚠️ double accolades dans l’exposant pour mathtext + f-string
plt.plot(
    V_plot, I_fit, lw=1.5,
    label=(
        fr"Courbe de Shockley"
    )
)

padx = 0.05*(df["V"].max() - df["V"].min() or 1.0)
pady = 0.05*(df["I"].max() - df["I"].min() or 1.0)
plt.xlim(df["V"].min()-padx, df["V"].max()+padx)
plt.ylim(df["I"].min()-pady, df["I"].max()+pady)

plt.xlabel("Tension V (V)")
plt.ylabel("Courant I (A)")
# plt.title("Courant i-v de la diode standard comparée au modèle de Shockley")
plt.legend(title = "Légende 1 - Courbes étudiées")
plt.tight_layout()
plt.savefig(OUT_PNG)
plt.show()

print(f"Paramètres ajustés : i0 = {i0:.3e} A, v0 = {v0:.4f} V (n·V_T)")
