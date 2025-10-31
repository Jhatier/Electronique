import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    array
        Array des puissances dissipées moyennes et de la résistance
    """
    
    r = arr[:, 0]   # Colonne de résistance
    v = arr[:, 1]   # Colonne de tension

    p_moy_dis = v**2 / r    # Array de la puissance moyenne dissipée

    return np.array([p_moy_dis, r])


def donnees_graphique(circuit):
    """
    Ressort un array de la résistance et de la puissance dissipée moyenne à utiliser pour le grpahique
    
    Paramètres
    circuit : str {"c" ou "f"}
        Indique si on travaille avec le circuit c ou f
    
    Retourne
    array
        array[0] donne la puissance moyenne dissipée moyenne et array[1] donne la résistance moyenne
    """
    print(circuit.lower(), type(circuit))
    if circuit.lower() != "c" and circuit.lower() != "f":
        raise TypeError("'circuit' doit être 'c' ou 'f'")
    
    r_moy = []
    p_moy_dis_moy = []

    for i in range(10):
        file = f"mesures_{circuit.lower()}_{i}.lvm"
        filepath = folder + file

        arr_puissance = puissance_moyenne_dissipee(read(filepath))

        r_moy.append(np.mean(arr_puissance[1]))
        p_moy_dis_moy.append(np.mean(arr_puissance[0]))
    
    return np.array([p_moy_dis_moy, r_moy])


def tracer_graphique(circuit):
    """
    Trace le graphique de la puissance moyenne dissipée selon la résistance. L'axe de la résistance est logarithmique
    
    Paramètres
    circuit : str {"c" ou "f"}
        Indique si on travaille avec le circuit c ou f
    """

    donnees = donnees_graphique(circuit)

    plt.semilogx(donnees[1], donnees[0])
    plt.xlim((0, 215))
    plt.ylim((0, np.max(donnees[0])) + np.max(donnees[0])*0.1)  # On va de 0 à 10% au-dessus de la valeur max en y
    plt.show()

tracer_graphique("f")
