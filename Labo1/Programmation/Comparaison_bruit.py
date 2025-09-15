import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


# Choisir le directory pour les figures et le créer s'il n'existe pas.
plot_dir = os.path.join("Labo1/Figures/signal_nul_vs_bruit_gaussien")
os.makedirs(plot_dir, exist_ok=True)

# Les fichiers de données
files = ['Labo1/Mesures/convertisseur_090925_01.lvm',               #0
         'Labo1/Mesures/convertisseur_débranché_100925_01.lvm',     #1
         'Labo1/Mesures/tension_patate_aluinox_090925_02.lvm',      #2
         'Labo1/Mesures/tension_patate_aluacier_90925_01.lvm',      #3
         'Labo1/Mesures/voltage_pile_100925_01.lvm',                #4
         'Labo1/Mesures/voltage_circuit_100925_01.lvm'              #5
         ]

# Descriptions et légendes
description = {0: 'le convertisseur',
               1: "le signal nul",
               2: "la pomme de terre avec une tige d'aluminium et d'inox",
               3: "la pomme de terre avec une tige d'aluminium et d'acier",
               4: "la pile",
               5: "le circuit"
               }

nom = {0: 'convertisseur',
       1: "signal nul",
       2: "aluminium - inox",
       3: "aluminium - acier",
       4: "pile",
       5: "circuit"
       }

num = 1
filepath = 'Labo1/Mesures/convertisseur_débranché_100925_01.lvm'

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy()
    col = 1
    if df.shape[1] == 3:
        col = 2
    arr = df.to_numpy()  
    if Path(file_name).name == "tension_patate_aluacier_90925_01.lvm":
        arr[:, 0] += 0.0085  
    return df.to_numpy()[:, :col]


def incertitude(array, indice):
    col = array[:, indice]
    return 0.5 * (np.max(col) - np.min(col))


def graphiques_scatter(array):
    fig = plt.gcf()
    fig.set_size_inches(10, 6)

    bruit = read(files[1])

    x_sig = np.arange(1, array.shape[0] + 1)
    x_brt = np.arange(1, bruit.shape[0] + 1)

    plt.plot(x_sig, array[:, 0], markersize=0.75, linestyle='none', marker='o', label=nom[num])
    plt.plot(x_brt,  bruit[:, 1], markersize=0.75, linestyle='none', marker='o', label="bruit gaussien")

    plt.errorbar(x_sig, array[:, 0], yerr=incertitude(array, 0), fmt='none',
                 elinewidth=0.6, capsize=1.5, alpha=0.6)

    plt.xlim(-5, 1015)
    plt.legend()
    plt.xlabel("Numéro d'index de la mesure")
    plt.ylabel("Tension [V]")

    plt.title(
        f"Fig. 7 - Tension mesurée lorsque le signal est nul et bruit gaussien généré par LabVIEW.",
        y=-0.15
    )

    plt.tight_layout()
    plt.savefig(plot_dir + "/signal_nul_vs_bruit.png")

    plt.show()


def histogramme(array):
    fig = plt.gcf()
    fig.set_size_inches(10, 6)

    plt.hist(array, bins=10, range=[-0.003, 0.005])

    plt.xlabel("Tension [V]")
    plt.ylabel("Nombre d'échantillons")
    plt.title("Fig. 6 - La distribution des valeurs de tensions mesurées pour le signal nul et la distribution du bruit" \
              " gaussien",
              y=-0.15)

    plt.tight_layout()
    plt.savefig(plot_dir + "/histogramme.png")

    plt.show()


histogramme(read(filepath))

# graphiques_scatter(read(filepath))
