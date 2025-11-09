import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Dossier où se trouvent les fichiers
folder = os.path.join("Labo5/Mesures/Partie2/")

# Dossier où les figures sont sauvegardées
plot_dir = os.path.join("Labo5/Figures/Partie2")
os.makedirs(plot_dir, exist_ok=True)

# Dictionnaire de ce qui est mesuré dans chaque canal pour chaque scope
mesures_canaux = {0: ["Module d'adaptation d'impédance", "Ligne de transmission"],
                  1: ["Module d'adaptation d'impédance", "Ligne de transmission"],
                  2: ["Module d'adaptation d'impédance"],
                  3: ["Module d'adaptation d'impédance", "Ligne de transmission"],
                  4: ["Module d'adaptation d'impédance"],
                  5: ["Module d'adaptation d'impédance", "Ligne de transmission"],
                  6: ["Module d'adaptation d'impédance", "Ligne de transmission"],
                   }

def read(filename):
    """
    Lire les fichiers csv sortis par l'oscilloscope.
    Première colonne : temps
    Deuxième colonne : canal 1
    Troisième colonne : canal 2
    
    Retourne un array numpy"""
    data = pd.read_csv(filename)

    data = data.to_numpy()[1:].astype(float)  # On enlève la première rangée et on mets les données en float
    return data

def graphique(scope):
    """
    Trace le graphique pour les différents scopes

    Paramètres
    scope : int {0 <= scope => 6}
        le numéro de scope pour notre nom de fichier
    """
    if scope not in range(7) or type(scope) != int:
        raise ValueError("scope doît être un int entre 0 et 6 inclusivement")

    data = read(folder + f"scope_{scope}.csv")
    temps = data[:, 0]

    plt.clf()
    fig = plt.figure(figsize=(8,5))

    # Pour chacun des canaux de l'oscilloscope enregistré sur le scope :
    for i in range(data.shape[1] - 1):
        canal = data[:, i+1]

        plt.plot(temps, canal, label=mesures_canaux[scope][i])

    plt.legend()
    plt.xlabel(r"Temps [s]")
    plt.ylabel(r"Tension [V]")
    plt.tight_layout()
    plt.savefig(plot_dir+f"scope_{scope}")

for i in range(7):
    graphique(i)
