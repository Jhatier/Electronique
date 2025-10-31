import numpy as np
import pandas as pd
import os


# Dossier où se trouvent les fichiers
folder = os.path.join("Labo5/Mesures/Partie1/")

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=21, decimal=",")
    df = df.iloc[:, 1:].copy()

    return df.to_numpy()[:, :2]

def puissance_moyenne_dissipee(arr):
    """
    Prend un array avec la tension DC RMS et la résistance et retourne un array de même taille avec p_moy dissipée

    Paramètres
    arr : array numpy
        La colonne 1 contient la résistance et la colonne deux contient la tension DC RMS
    
    Retourne
    p_moy_dis : array
        Array des puissances dissipées
    """
    
    r = arr[:, 0]   # Colonne de résistance
    v = arr[:, 1]   # Colonne de tension

    p_moy_dis = v**2 / r    # Array de la puissance moyenne dissipée

    return p_moy_dis
