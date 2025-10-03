import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io, re

FOR_FILE = Path("Labo3/Mesures/914b_donnees_iv_01.lvm")     # direct 0→1 V
REV_FILE = Path("Labo3/Mesures/914binv_donnees_iv_01.lvm")  # inverse 0→6 V
FIG_DIR  = Path("Labo3/Figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG  = FIG_DIR / "diode_iv_combinee.png"

# --- Lecture flexible: trouve la 1re ligne numérique, essaie décimales , puis . ---
def read_lvm_auto(path: Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()
    num_re = re.compile(r"^\s*[-+]?(?:\d+|\d*[.,]\d+)\s")
    start = next(i for i,ln in enumerate(lines) if num_re.match(ln))
    block = "\n".join(lines[start:])
    for dec in [",", "."]:
        df = pd.read_csv(io.StringIO(block), sep=r"[\t;]+", engine="python", decimal=dec, header=None)
        num = df.select_dtypes("number")
        if num.shape[1] >= 2:
            num = num.iloc[:, :2].copy()
            a, b = num.iloc[:,0].to_numpy(), num.iloc[:,1].to_numpy()
            def mono_score(x):
                d = np.diff(x); 
                return (d>0).mean() if d.size else 0.0
            if mono_score(b) > mono_score(a):
                V, I = b, a
            else:
                V, I = a, b
            out = pd.DataFrame({"V": V.astype(float), "I": I.astype(float)}).dropna()
            return out
    return pd.DataFrame(columns=["V","I"])

df_for = read_lvm_auto(FOR_FILE) 
df_rev = read_lvm_auto(REV_FILE)


df_for["V"] =  np.abs(df_for["V"])
if df_for["I"].median() < 0:
    df_for["I"] = -df_for["I"]

df_rev["V"] = -np.abs(df_rev["V"])

df = pd.concat([df_rev, df_for], ignore_index=True).sort_values("V").reset_index(drop=True)

plt.figure(figsize=(9,6))
plt.plot(df["V"], df["I"], marker='o', markersize=3, linewidth=1)

padx = 0.05*(df["V"].max() - df["V"].min() or 1.0)
pady = 0.05*(df["I"].max() - df["I"].min() or 1.0)
plt.xlim(df["V"].min()-padx, df["V"].max()+padx)
plt.ylim(df["I"].min()-pady, df["I"].max()+pady)

plt.xlabel("V_d (V)")
plt.ylabel("I_d (A)")
plt.title("Diode standard — courbe i–v combinée (inverse + direct)")
plt.tight_layout()
plt.savefig(OUT_PNG)
plt.show()
