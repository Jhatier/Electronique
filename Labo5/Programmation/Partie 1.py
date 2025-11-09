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
    v = arr[:, 1] / np.sqrt(10)  # Colonne de tension

    p_moy_dis = (v)**2 / r    # Array de la puissance moyenne dissipée

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


def resistance_inc(values: np.ndarray):
    """
    Calcule la valeur de l'incertitude sur la résistance selon le fabricant
    
    Paramètres
    values : np.ndarray
        Valeurs mesurées de la résistance

    Retourne
    incertitudes : np.ndarray
        Array des incertitudes
    """
    incertitudes = np.zeros_like(values, dtype=float)
    inc100 = values <= 100
    incertitudes[inc100] = 0.00003 * values[inc100] + 0.00003 * 0.001
    inc1k = (values > 100) & (values <= 1000)
    incertitudes[inc1k] = 0.00002 * values[inc1k] + 0.000005 * 0.001
    inc10k = (values > 1000) & (values <= 10000)
    incertitudes[inc10k] = 0.00002 * values[inc10k] + 0.000005 * 0.0001
    inc100k = (values > 10000) & (values <= 100000)
    incertitudes[inc100k] = 0.00002 * values[inc100k] + 0.000005 * 0.00001
    inc1M = (values > 100000) & (values <= 1000000)
    incertitudes[inc1M] = 0.00002 * values[inc1M] + 0.00001 * 0.000005
    inc10M = (values > 1000000) & (values <= 10000000)
    incertitudes[inc10M] = 0.00015 * values[inc10M] + 0.00001 * 0.0000005
    inc100M = (values > 10000000) & (values <= 100000000)
    incertitudes[inc100M] = 0.003 * values[inc100M] + 0.0001 * 0.0000005

    return incertitudes


def puissance_inc(values):
    """
    values : array
        l'array obtenu avec read(file)
    """
    unique = np.unique(values[:, 1])
    try:
        incertitude = unique[1] - unique[0]
    except:
        incertitude = 0.01

    return np.mean(incertitude**2 / values[:, 0])


def incertitude_graphique(circuit):
    """
    Ressort un array qui donne l'erreur en x et en y pour chaque point du graphique avec l'écart-type.
    
    Paramètres
    circuit : str {"c" ou "f"}
        Indique si on travaille avec le circuit c ou f
    sigma : float
        Le nombre d'écarts-types utilisé pour trouver l'incertitude.
    
    Retourne
    array
        array[0] donne l'incertitude sur la puissance moyenne dissipée moyenne et array[1] donne l'incertitude sur la
        résistance moyenne
    """
    if circuit.lower() != "c" and circuit.lower() != "f":
        raise TypeError("'circuit' doit être 'c' ou 'f'")
    
    r_err = []
    p_moy_dis_err = []

    for i in range(10):
        file = f"mesures_{circuit.lower()}_{i}.lvm"
        filepath = folder + file

        arr_puissance = puissance_moyenne_dissipee(read(filepath))

        r_err.append(np.mean(resistance_inc(arr_puissance[1])))
        p_moy_dis_err.append(puissance_inc(read(filepath)))
    
    return np.array([p_moy_dis_err, r_err])


def fonction_theorique(circuit, V=1, r_s=50, lim=(0,215)):
    """
    On rentre les bornes en x et le circuit pour lequel on veut faire la fonction et ça retourne un array de la courbe théorique.
    
    Paramètres
    circuit : str {"c" ou "f"}
        Indique si on travaille avec le circuit c ou f
    V : float
        Tension à la source
    r_s : float
        Résistance de la source
    lim : tuple
        les limites dans l'axe x
    
    Retourne
    array
        un array 2D de la puissance dissipée théorique selon la résistance
    """
    if circuit.lower() != "c" and circuit.lower() != "f":
        raise TypeError("'circuit' doit être 'c' ou 'f'")
    
    if circuit == 'c':
        r_ch = np.linspace(lim[0], lim[1], 1000)

        puissance = (r_ch * V**2) / (2 * (r_ch + r_s)**2)

        return np.array([puissance, r_ch])
    
    if circuit == 'f':
        # Définition de constantes utiles
        w = 2000 * np.pi    # 2pi*f où f=1kHz
        C = 4e-6            # La capacitance du condensateur
        w2C2 = (w * C) ** 2 # Le carré du produit de w et C.

        r_ch = np.linspace(lim[0], lim[1], 1000)

        numérateur = r_ch * V**2 * (r_ch**2 + w2C2)**2          # C lè
        dénominateur = 2 * (r_ch**2 * w2C2**2 + r_ch**4 * w2C2) # vréman lè

        puissance = numérateur / dénominateur

        plt.plot(r_ch, puissance)
        plt.show()

        return np.array([puissance, r_ch])
        
# fonction_theorique("f")

def tracer_graphique(circuit):
    """
    Trace le graphique de la puissance moyenne dissipée selon la résistance. L'axe de la résistance est logarithmique
    
    Paramètres
    circuit : str {"c" ou "f"}
        Indique si on travaille avec le circuit c ou f
    sigma : float
        Le nombre d'écarts-types utilisé pour trouver l'incertitude.
    """
    limites_x = (10, 215)

    donnees = donnees_graphique(circuit)
    incertitude = incertitude_graphique(circuit)

    puissance = fonction_theorique(circuit, 1, 50, limites_x)

    plt.errorbar(donnees[1], donnees[0], xerr=incertitude[1], yerr=incertitude[0], linestyle='none',
                 marker='o', markersize=3)
    plt.plot(puissance[1], puissance[0])
    plt.xscale('log')
    plt.xlim(limites_x[0], limites_x[1])
    plt.ylim(0, np.max(donnees[0]) + np.max(donnees[0])*0.1)  # On va de 0 à 10% au-dessus de la valeur max en y
    plt.xlabel(r"Résistance [$\Omega$]")
    plt.ylabel(r"Puissance moyenne dissipée [$W$]")
    plt.show()

tracer_graphique("f")
