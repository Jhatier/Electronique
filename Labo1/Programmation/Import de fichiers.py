import pandas as pd
import matplotlib.pyplot as plt
import numpy as np    

#Labo1/Mesures/tension_patate_aluacier_90925_01.lvm
#Labo1/Mesures/tension_patate_aluacier_90925_02.lvm
#Labo1/Mesures/tension_patate_aluacier_90925_03.lvm
#Labo1/Mesures/tension_patate_aluinox_090925_01.lvm
#Labo1/Mesures/tension_patate_aluinox_090925_02.lvm
#Labo1/Mesures/voltage_circuit_090925_01.lvm
#Labo1/Mesures/voltage_pile_090925_01.lvm


filepath = "Labo1/Mesures/convertisseur_débranché_090925_01.lvm"


def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy() 

    return df.to_numpy()[:, :2]

print(read(filepath))

y = read(filepath)[:, 0]
x = read(filepath)[:, 1]

plt.figure()
plt.plot(x, y, linewidth=1)
plt.xlabel("Bruit gaussien")
plt.ylabel("Amplitude [V]")
plt.title("Amplitude de la différence de potentiel aux bornes du convertisseur")
plt.tight_layout()
plt.show()