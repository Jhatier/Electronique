import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Choisir le directory pour les figures et le créer s'il n'existe pas.
plot_dir = os.path.join("Labo1/Figures/tension_constante")
os.makedirs(plot_dir, exist_ok=True)

# Les fichiers de données
files = ['Labo1/Mesures/convertisseur_090925_01.lvm',               #0
         'Labo1/Mesures/convertisseur_débranché_090925_01.lvm',     #1    
         'Labo1/Mesures/convertisseur_débranché_090925_02.lvm',     #2
         'Labo1/Mesures/convertisseur_débranché_100925_01.lvm',     #3
         'Labo1/Mesures/tension_patate_aluinox_090925_01.lvm',      #4
         'Labo1/Mesures/tension_patate_aluinox_090925_02.lvm',      #5
         'Labo1/Mesures/tension_patate_aluacier_90925_01.lvm',      #6
         'Labo1/Mesures/tension_patate_aluacier_90925_02.lvm',      #7
         'Labo1/Mesures/tension_patate_aluacier_90925_03.lvm',      #8
         'Labo1/Mesures/voltage_pile_090925_01.lvm',                #9
         'Labo1/Mesures/voltage_pile_100925_01.lvm',                #10
         'Labo1/Mesures/voltage_circuit_090925_01.lvm',             #11
         'Labo1/Mesures/voltage_circuit_100925_01.lvm'              #12
         ]

# Descriptions et légendes
description = {0: 'le convertisseur',
        4: "la pomme de terre avec une tige d'aluminium et d'inox",
        5: "la pomme de terre avec une tige d'aluminium et d'inox",
        6: "la pomme de terre avec une tige d'aluminium et d'acier",
        7: "la pomme de terre avec une tige d'aluminium et d'acier",
        8: "la pomme de terre avec une tige d'aluminium et d'acier",
        9: "la pile",
        10: "la pile",
        11: "le circuit",
        12: "le circuit"
        }

nom = {0: 'convertisseur',
       4: "aluminium - inox",
       5: "aluminium - inox",
       6: "aluminium - acier",
       7: "aluminium - acier",
       8: "aluminium - acier",
       9: "pile",
       10: "pile",
       11: "circuit",
       12: "circuit"
       }

num = 0     # L'index du fichier utilisé.
filepath = files[num]

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy()
    col = 1
    if df.shape[1] == 3:
        col = 2
    return df.to_numpy()[:, :col]

# --- FIX: stats sur la colonne (et non la ligne) ---
def moyenne(file_name, indice):
    return float(np.mean(read(file_name)[:, indice]))

def variance(file_name, indice):
    return float(np.var(read(file_name)[:, indice], ddof=0))

def snr(file_name, indice):
    m = moyenne(file_name, indice); v = variance(file_name, indice)
    return (m*m)/v if v != 0 else np.inf

def incertitude(array, indice):
    col = array[:, indice]
    return 0.5 * (np.max(col) - np.min(col))

def graphiques_scatter(array):
    fig = plt.gcf()
    fig.set_size_inches(10, 6)

    bruit = read(files[3])

    # --- FIX: abscisses propres à chaque série ---
    x_sig = np.arange(1, array.shape[0] + 1)
    x_brt = np.arange(1, bruit.shape[0] + 1)

    plt.plot(x_sig, array[:, 0], markersize=0.75, linestyle='none', marker='o', label=nom[num])
    plt.plot(x_brt,  bruit[:, 0], markersize=0.75, linestyle='none', marker='o', label="signal nul")

    plt.errorbar(x_sig, array[:, 0], yerr=incertitude(array, 0), fmt='none',
                 elinewidth=0.6, capsize=1.5, alpha=0.6)

    plt.xlim(-5, 1015)
    plt.legend()
    plt.xlabel("Numéro d'index de la mesure")
    plt.ylabel("Tension [V]")

    plt.title(
        f"Fig. 1 La tension dans {description[num]} et la tension mesurée lorsque\nle signal est nul."
        f"\nLes barres d'incertitudes sur le signal nul sont présentes, mais difficilement visibles.",
        y=-0.25
    )

    ax = plt.gca()
    ax.text(0.05, 0.88, f"v = {moyenne(filepath, 0):.3f}", transform=ax.transAxes)
    ax.text(0.05, 0.83, f"s^2 = {variance(filepath, 0):.3f}", transform=ax.transAxes)
    ax.text(0.05, 0.78, f"SNR = {snr(filepath, 0):.3f}", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"_{nom[num]}.png"))

graphiques_scatter(read(filepath))
