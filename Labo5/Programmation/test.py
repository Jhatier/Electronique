import os
import numpy as np
import pandas as pd

path = os.path.join("Labo5/Mesures/Partie1/mesures_c_")

def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=21, decimal=",")
    df = df.iloc[:, 1:].copy()

    return df.to_numpy()[:, :2]


for i in range(10):
    end = f"{i}.lvm"

    data = read(path+end)

    V = np.std(data[:, 0]/10)
    R = np.std(data[:, 1])

    print(f"{i}\nTension : {V}\nRésistance : {R}")
