import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Choisir le directory pour les figures et le créer s'il n'existe pas.
plot_dir = os.path.join("Labo1/Figures/tension_constante")
os.makedirs(plot_dir, exist_ok=True)

# Les fichiers de données dans l'ordres demandés par le protocoles. Cela permet de simplement appeler files[4] par ex.
files = ['Labo1/Mesures/convertisseur_090925_01.lvm',               #0
         'Labo1/Mesures/convertisseur_débranché_100925_01.lvm',     #1
         'Labo1/Mesures/tension_patate_aluinox_090925_02.lvm',      #2
         'Labo1/Mesures/tension_patate_aluacier_90925_01.lvm',      #3
         'Labo1/Mesures/voltage_pile_100925_01.lvm',                #4
         'Labo1/Mesures/voltage_circuit_100925_01.lvm'              #5
         ]


num = 5     # L'index du fichier utilisé.
filepath = files[num]

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy()

    col = 1 # On ne prend qu'une seule colonne si nous n'avons qu'une seule colonne de données.
    if df.shape[1] == 3:    # On prend 2 colonnes si on a deux colonnes de données (3 colonnes données par pandas).
        col = 2

    return df.to_numpy()[:, :col]

def extraction_colonne(array, indice):
    return array[:, indice].tolist()


distribution_résistance = (lambda a: 12*a[:,0]/a[:,1])(read(filepath))

#print(np.average(distribution_résistance))      # Moyenne de la distribution
#print(np.median(distribution_résistance))   # Médiane de la distribution
#print((np.average(distribution_résistance)-1000)/10)    # Pourcentage d'écart
#print(np.std(distribution_résistance))  # Écart-type
#print(np.std(distribution_résistance)/np.mean(distribution_résistance))

def moyenne(file_name, indice):
    return float(np.mean(read(file_name)[:, indice]))

def variance(file_name, indice):
    return float(np.var(read(file_name)[:, indice], ddof=0))

def snr(file_name, indice):
    m = moyenne(file_name, indice); v = variance(file_name, indice)
    return (m*m)/v if v != 0 else np.inf


for i in range (0, 4):
    print(snr(files[i], 0))

def incertitude(array, indice):
    col = array[:, indice]
    return 0.5 * (np.max(col) - np.min(col))



import numpy as np
from math import isnan

# ---------- Outils Type A ----------
# Table t de Student à 95 % bilatéral (alpha=0.05), df=1..30 (fallback si SciPy absent)
_T_TABLE_95 = {
    1:12.706, 2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262, 10:2.228,
    11:2.201, 12:2.179, 13:2.160, 14:2.145, 15:2.131, 16:2.120, 17:2.110, 18:2.101, 19:2.093, 20:2.086,
    21:2.080, 22:2.074, 23:2.069, 24:2.064, 25:2.060, 26:2.056, 27:2.052, 28:2.048, 29:2.045, 30:2.042
}

def _student_k(dof: int, conf: float = 0.95) -> float:
    """
    Renvoie le facteur k (quantile de Student à conf bilatéral) pour dof degrés de liberté.
    Utilise SciPy si dispo; sinon, table 95% + approximation normale si dof>30.
    """
    alpha = 1.0 - conf
    # Essai SciPy si présent
    try:
        from scipy.stats import t
        return float(t.ppf(1 - alpha/2, dof))
    except Exception:
        pass
    # Fallback sans SciPy (95% seulement)
    if abs(conf - 0.95) < 1e-6:
        if dof <= 30:
            return _T_TABLE_95.get(max(1, dof))
        else:
            return 1.96  # approx. normale pour df > 30
    # Autres niveaux de confiance sans SciPy: approximation normale
    from math import erf, sqrt
    # approx bilatérale via inverse erf ≈ binaire simple
    # niveaux classiques: 0.90→1.645, 0.95→1.96, 0.99→2.576
    if conf >= 0.99: return 2.576
    if conf >= 0.95: return 1.96
    if conf >= 0.90: return 1.645
    return 1.96

def incertitude_type_A(x, conf: float = 0.95):
    """
    x : array-like des répétitions d'une grandeur (1D).
    Renvoie un dict avec moyenne, s (Bessel), u_A, k, U, n, dof.
    Ignore les NaN automatiquement.
    """
    x = np.asarray(x).astype(float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        raise ValueError("Au moins 2 valeurs non-NaN sont requises pour estimer un écart-type (type A).")
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))      # écart-type expérimental (Bessel)
    uA = s / np.sqrt(n)               # incertitude type de la moyenne
    dof = n - 1
    k = _student_k(dof, conf)
    U = k * uA                        # incertitude élargie pour le niveau 'conf'
    return {
        "n": n,
        "dof": dof,
        "mean": mean,
        "s": s,
        "uA": uA,
        "k": k,
        "U": U,
        "conf": conf
    }

def imprimer_rapport_type_A(res, unite: str = ""):
    """ jolis prints pour le dict retourné par incertitude_type_A """
    u = f" {unite}" if unite else ""
    print(f"n = {res['n']}, dof = {res['dof']}, niveau de confiance ≈ {int(res['conf']*100)} %")
    print(f"moyenne = {res['mean']:.6g}{u}")
    print(f"écart-type (Bessel) s = {res['s']:.6g}{u}")
    print(f"u_A (écart-type de la moyenne) = {res['uA']:.6g}{u}")
    print(f"k (Student) = {res['k']:.4g}")
    print(f"U = k·u_A = {res['U']:.6g}{u}")
    print(f"→ Résultat (≈{int(res['conf']*100)} %) : {res['mean']:.6g} ± {res['U']:.6g}{u}")


# distribution_résistance est déjà calculée dans ton script :
# distribution_résistance = (lambda a: 12*a[:,0]/a[:,1])(read(filepath))

res = incertitude_type_A(distribution_résistance, conf=0.95)
imprimer_rapport_type_A(res, unite="Ω")   # change l’unité si besoin

# Si tu veux aussi le 1σ (non élargi), c'est res["uA"] :
print("À 1σ (type) :", f"{res['mean']:.6g} ± {res['uA']:.6g} Ω")