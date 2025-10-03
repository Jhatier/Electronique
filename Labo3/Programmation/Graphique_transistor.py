import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io, re, math

DATA_FILE = Path("Labo3/Mesures/transistor_donnees_iv_01.lvm")
OUT_PNG = Path("Labo3/Figures/transistor_ic_vc_ipla.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

def read_lvm(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    num_re = re.compile(r"^\s*[-+]?(?:\d+|\d*[.,]\d+)\s")
    start = next(i for i,l in enumerate(lines) if num_re.match(l))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])),
                     sep=r"\t|;", engine="python", decimal=",", header=None)
    df = df.select_dtypes("number")
    if df.shape[1] >= 3:
        df = df.iloc[:, :3]; df.columns = ["Vc","Ic","Vb"]
    else:
        df = df.iloc[:, :2]; df.columns = ["Vc","Ic"]
    return df

def group_runs(df: pd.DataFrame):
    if "Vb" in df.columns:
        return {float(vb): sub.sort_values("Vc") for vb, sub in df.groupby("Vb")}
    vc = df["Vc"].to_numpy()
    idx = [0]
    for i in range(1, len(vc)):
        if vc[i] < 0.2*vc[i-1]:
            idx.append(i)
    idx.append(len(df))
    return {k: df.iloc[idx[k]:idx[k+1]].sort_values("Vc") for k in range(len(idx)-1) if idx[k+1]-idx[k] > 2}

I_EPS = 5e-4       # 0.5 mA : clipping visuel
ZERO_THRESH = 1e-3 # 1 mA : seuil pour "≈ 0 A" (annotation unique)

def baseline_correct(sub: pd.DataFrame, clip=True):
    v = sub["Vc"].to_numpy(); y = sub["Ic"].to_numpy()
    w = v <= (v.min() + 0.15*(v.max()-v.min()))
    off = np.median(y[w]) if np.any(w) else np.median(y[:max(3, len(y)//10)])
    yc = y - off
    if clip:
        yc[np.abs(yc) < I_EPS] = 0.0  # clip visuel uniquement
    out = sub.copy(); out["Ic"] = yc
    return out

def plateau_points(sub: pd.DataFrame):
    x = sub["Vc"].to_numpy(); y = sub["Ic"].to_numpy()
    dx = np.gradient(x); dy = np.gradient(y)
    slope = np.abs(dy/np.where(dx==0, 1, dx))
    mask_x = x >= (x.min() + 0.7*(x.max()-x.min()))
    thr = 0.05*np.nanmax(np.abs(y))/(x.max()-x.min()+1e-12)
    flat = (slope < thr) & mask_x
    if np.count_nonzero(flat) >= 3:
        return y[flat]
    k = max(3, int(0.2*len(sub)))
    return y[-k:]

def ic95_student(sub):
    pts = plateau_points(sub)
    N = len(pts)
    m = float(np.mean(pts))
    s = float(np.std(pts, ddof=1)) if N >= 2 else 0.0
    t = 1.96 if N>30 else {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}.get(N-1,2.0)
    hw = t*s/np.sqrt(N) if N>=2 else 0.0
    return m, hw

def one_sig_unc(unc):
    if unc == 0: return 0.0, 0
    exp = math.floor(math.log10(abs(unc)))
    lead = abs(unc) / (10**exp)
    lead_r = round(lead)
    if lead_r == 10:
        lead_r = 1
        exp += 1
    return lead_r * (10**exp), exp

def choose_unit(unc_or_val):
    a = abs(unc_or_val) if unc_or_val != 0 else 1.0
    for scale, unit in [(1.0,"A"), (1e3,"mA"), (1e6,"µA"), (1e9,"nA")]:
        if 1 <= a*scale < 1000:
            return scale, unit
    return 1.0, "A"

def fmt_val_unc(val, unc):
    base = unc if unc!=0 else (abs(val) if val!=0 else 1.0)
    scale, unit = choose_unit(base)
    v = val*scale; u = abs(unc)*scale
    u1, exp = one_sig_unc(u)
    dec = max(0, -exp)
    v_r = round(v, dec)
    fmt = f"{{:.{dec}f}}"
    return f"{fmt.format(v_r)} ± {fmt.format(u1)} {unit}"

def fmt_vb(v):
    return f"{v:.1f} V".replace('.', ',')

df = read_lvm(DATA_FILE)
groups = group_runs(df)

palette = plt.cm.tab10.colors if len(groups) <= 10 else plt.cm.tab20.colors
if "Vb" not in df.columns:
    n = len(groups)
    vb_sequence = [0.2*(i+1) for i in range(n)]  # 0.2, 0.4, ..., 1.2 si n=6
else:
    vb_sequence = None

plt.figure(figsize=(10,6))
zero_done = False

for i, (key, sub_raw) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
    # tracer: version clippée ; incertitude: version non clippée
    sub_clip   = baseline_correct(sub_raw, clip=True)
    sub_noclip = baseline_correct(sub_raw, clip=False)

    x = sub_clip["Vc"].to_numpy(); y = sub_clip["Ic"].to_numpy()
    color = palette[i % len(palette)-1]

    label = fmt_vb(key) if "Vb" in df.columns else fmt_vb(vb_sequence[i])
    plt.plot(x, y, marker='o', markersize=3, linewidth=1.2, label=label, color=color)

    m, hw = ic95_student(sub_noclip)

    dx = 0.04*(x.max()-x.min() or 1.0)
    dy = 0.04*(y.max()-y.min() or 1.0)
    x_annot = x.max() - dx

    if abs(m) < ZERO_THRESH:
        if not zero_done:
            plt.annotate(f"i_pla ≈ {fmt_val_unc(0.0, hw)}",
                         xy=(x_annot, 0.0),
                         xytext=(x_annot - 2.5*dx, 0.0 - 0.75*dy),
                         ha="left", va="bottom", color="black")
            zero_done = True
    else:
        plt.annotate(f"i_pla ≈ {fmt_val_unc(m, hw)}",
                     xy=(x_annot, m),
                     xytext=(x_annot - 2.5*dx, m - 3.5*dy),
                     ha="left", va="bottom", color="black")

plt.xlabel("Tension (v_C) [V]")
plt.ylabel("Courant (i_C) [A]")
plt.title("Courant passant dans le transistor en fonction de la tension appliquée")
plt.legend(title="Légende 1 - Valeur de V_b", ncol=2, frameon=True)
plt.tight_layout()
plt.savefig(OUT_PNG)
plt.show()
