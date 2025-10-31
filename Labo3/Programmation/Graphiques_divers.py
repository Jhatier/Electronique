import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


plot_dir = os.path.join("Labo3/Figures/")
os.makedirs(plot_dir, exist_ok=True)

files = ["Labo3/Mesures/resistance_donnees_iv_01.lvm",    # 0
         "Labo3/Mesures/transistor_donnees_iv_01.lvm",    # 1
         "Labo3/Mesures/condensateur_donnees_iv_01.lvm",  # 2
         "Labo3/Mesures/bobine_donnees_iv_01.lvm",        # 3
         "Labo3/Mesures/914b_donnees_iv_01.lvm",          # 4
         "Labo3/Mesures/914binv_donnees_iv_01.lvm",       # 5
         "Labo3/Mesures/zener_donnees_iv_01.lvm"          # 6
         ]

num = 5     # Numéro du fichier utilisé
filepath = files[num]

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy()

    # col = 1 # On ne prend qu'une seule colonne si nous n'avons qu'une seule colonne de données.
    # if df.shape[1] == 3:    # On prend 2 colonnes si on a deux colonnes de données (3 colonnes données par pandas).
    #     col = 2

    return df.to_numpy()[:, :2]

data = read(filepath)

print(data)

# plt.plot(data[:, 0], data[:, 1])
# plt.xlabel("Tension")
# plt.ylabel("Courant")
# plt.title(f"Courbe i-v de {filepath.split("/")[-1].split("_")[0]}")
# plt.savefig(plot_dir + f"courbe_i-v_{filepath.split("/")[-1].split("_")[0]}.png")
# plt.close()
