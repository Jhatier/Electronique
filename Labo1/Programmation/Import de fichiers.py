import pandas as pd
import matplotlib.pyplot as plt
import numpy as np    


# Les fichiers de données dans l'ordres demandés par le protocoles. Cela permet de simplement appeler files[4] par ex.
files = ['Labo1/Mesures/convertisseur_090925_01.lvm',
         'Labo1/Mesures/convertisseur_débranché_090925_01.lvm',
         'Labo1/Mesures/convertisseur_débranché_090925_02.lvm',
         'Labo1/Mesures/convertisseur_débranché_100925_01.lvm',
         'Labo1/Mesures/tension_patate_aluinox_090925_01.lvm',
         'Labo1/Mesures/tension_patate_aluinox_090925_02.lvm',
         'Labo1/Mesures/tension_patate_aluacier_90925_01.lvm',
         'Labo1/Mesures/tension_patate_aluacier_90925_02.lvm',
         'Labo1/Mesures/tension_patate_aluacier_90925_03.lvm',
         'Labo1/Mesures/voltage_pile_090925_01.lvm',
         'Labo1/Mesures/voltage_pile_100925_01.lvm',
         'Labo1/Mesures/voltage_circuit_090925_01.lvm',
         'Labo1/Mesures/voltage_circuit_100925_01.lvm'
         ]

filepath = "Labo1/Mesures/convertisseur_débranché_090925_01.lvm"
filepath = files[0]

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy()

    col = 1 # On ne prend qu'une seule colonne si nous n'avons qu'une seule colonne de données.
    if df.shape[1] == 3:    # On prend 2 colonnes si on a deux colonnes de données (3 colonnes données par pandas).
        col = 2

    return df.to_numpy()[:, :col]


def graphiques_scatter(array):
    plt.clf()
    bruit = read(files[3])

    plt.plot(np.linspace(1, array.shape[0], array.shape[0]), array[:, 0], markersize=0.75, linestyle='none',
             marker='o')
    plt.plot(np.linspace(1, array.shape[0], array.shape[0]), bruit[:, 0], markersize=0.75, linestyle='none',
             marker='o')
    plt.show()

graphiques_scatter(read(filepath))

# y = read(filepath)[:, 0]
# x = read(filepath)[:, 1]

# plt.figure()
# plt.plot(x, y, linewidth=1)
# plt.xlabel("Bruit gaussien")
# plt.ylabel("Amplitude [V]")
# plt.title("Amplitude de la différence de potentiel aux bornes du convertisseur")
# plt.tight_layout()
# plt.show()
