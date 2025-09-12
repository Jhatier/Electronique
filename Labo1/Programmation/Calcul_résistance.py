import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Choisir le directory pour les figures et le créer s'il n'existe pas.
plot_dir = os.path.join("Labo1/Figures/tension_constante")
os.makedirs(plot_dir, exist_ok=True)

# Les fichiers de données dans l'ordres demandés par le protocoles. Cela permet de simplement appeler files[4] par ex.
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


num = 12     # L'index du fichier utilisé.
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
print(np.average(distribution_résistance))      # Moyenne de la distribution
print(np.median(distribution_résistance))   # Médiane de la distribution
print((np.average(distribution_résistance)-1000)/10)    # Pourcentage d'écart 

def incertitude(array, indice):
    col = array[:, indice]
    return 0.5 * (np.max(col) - np.min(col))
