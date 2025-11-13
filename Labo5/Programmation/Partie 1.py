import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from scipy.optimize import curve_fit


# Dossier où se trouvent les fichiers
folder = os.path.join("Labo5/Mesures/Partie1/")

# Dossier pour plot les graphiques
plot_dir = os.path.join("Labo5/Figures/Partie1")
os.makedirs(plot_dir, exist_ok=True)

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
    
    r = arr[:, 1]       # Colonne de résistance
    v = arr[:, 0] / 10  # Colonne de tension

    p_moy_dis = (v)**2 / r    # Array de la puissance moyenne dissipée

    # Calcul des incertitudes
    unique = np.unique(v)
    try:
        inc_v = unique[1] - unique[0]
    except:
        inc_v = 0.001
    inc_r = resistance_inc(np.mean(r))
    # inc_p = np.mean(np.sqrt(p_moy_dis**2 * (2 * (inc_v/v)**2 + (inc_r/r)**2))) # Devrait être l'incertitude
    inc_p = np.mean((abs(2 * v * inc_v - v**2 * inc_r)) / r**2) # Devrait être l'incertitude

    return np.array([p_moy_dis, r]), [inc_p, inc_r]


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

        arr_puissance, _ = puissance_moyenne_dissipee(read(filepath))

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
    incertitudes[inc100] = 0.00003 * values[inc100] + 0.00003 * 100
    inc1k = (values > 100) & (values <= 1000)
    incertitudes[inc1k] = 0.00002 * values[inc1k] + 0.000005 * 1000
    inc10k = (values > 1000) & (values <= 10000)
    incertitudes[inc10k] = 0.00002 * values[inc10k] + 0.000005 * 10000
    inc100k = (values > 10000) & (values <= 100000)
    incertitudes[inc100k] = 0.00002 * values[inc100k] + 0.000005 * 100000
    inc1M = (values > 100000) & (values <= 1000000)
    incertitudes[inc1M] = 0.00002 * values[inc1M] + 0.00001 * 1000000
    inc10M = (values > 1000000) & (values <= 10000000)
    incertitudes[inc10M] = 0.00015 * values[inc10M] + 0.00001 * 10000000
    inc100M = (values > 10000000) & (values <= 100000000)
    incertitudes[inc100M] = 0.003 * values[inc100M] + 0.0001 * 100000000

    return incertitudes


def incertitude_graphique(circuit):
    """
    Ressort un array qui donne l'erreur en x et en y.
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

        arr_puissance, inc = puissance_moyenne_dissipee(read(filepath))

        r_err.append(inc[1])# np.mean(resistance_inc(arr_puissance[1])))
        p_moy_dis_err.append(inc[0])
    
    return np.array([p_moy_dis_err, r_err])


def fonction_theorique_c(r_ch, V=1, r_s=50):
    puissance = (r_ch * V**2) / (2 * (r_ch + r_s)**2)

    return puissance


def fonction_theorique_f(r_ch, V=1, r_s=50, wC=0.02513):
    # Définition de constantes utiles
    w = 2000 * np.pi    # 2pi*f où f=1kHz

    numérateur = r_ch * V**2                                        # C lè
    dénominateur = 2 * ((r_s + r_ch)**2 + (r_s * r_ch * wC)**2)  # vréman lè

    puissance = numérateur / dénominateur

    return puissance

def fit(circuit):
    data = donnees_graphique(circuit)
    if circuit == 'f':
        param, param_cov = curve_fit(fonction_theorique_f, data[1], data[0],
                                     bounds=([0.96, 45, 0.01508], [1.04, 55, 0.03519]))

        r = np.linspace(10, 215, 1000)
        w = 2000 * np.pi    # 2pi*f où f=1kHz
        num = r * param[0]**2
        dénom = 2 * ((param[1] + r)**2 + (param[1] * r * param[2])**2)
        puissance = num/dénom

    if circuit == 'c':
        param, param_cov = curve_fit(fonction_theorique_c, data[1], data[0])

        r = np.linspace(10, 215, 1000)
        puissance = (r * param[0]**2) / (2 * (r + param[1])**2)

    return puissance, param, param_cov

# curve, _, _, _ = fit('c')
# maxi = np.argmax(curve)
# print(np.linspace(10, 215, 1000)[maxi])

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

    curve = fit(circuit)
    puissance = curve[0]
    if circuit == "f":
        V, r_s, C = curve[1][0], curve[1][1], curve[1][2]
        errs = np.sqrt(np.diag(curve[2]))
        inc_V, inc_r_s, inc_C = errs[0], errs[1], errs[2]

        print(V, inc_V)
        print(r_s, inc_r_s)
        print(C, inc_C)
    elif circuit == "c":
        V, r_s = curve[1][0], curve[1][1]
        errs = np.sqrt(np.diag(curve[2]))
        inc_V, inc_r_s = errs[0], errs[1]

        print(V, inc_V)
        print(r_s, inc_r_s)

    plt.clf()

    fig = plt.figure(figsize=(8,5))

    plt.grid(True, which='both')

    plt.errorbar(donnees[1], donnees[0], xerr=incertitude[1], yerr=incertitude[0], linestyle='none',
                 marker='o', markersize=3, label='Puissance (données expérimentales)')
    plt.plot(np.linspace(limites_x[0], limites_x[1], 1000), puissance, label="Puissance (courbe théorique lissée avec scipy)")

    plt.xscale('log')
    plt.xlim(limites_x[0], limites_x[1])
    plt.ylim(0, np.max(donnees[0]) + np.max(donnees[0])*0.15)  # On va de 0 à 10% au-dessus de la valeur max en y

    plt.xlabel(r"Résistance [$\Omega$]")
    plt.ylabel(r"Puissance moyenne dissipée [W]")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_dir+f"/Circuit_{circuit}.png")

tracer_graphique("f")
tracer_graphique("c")
