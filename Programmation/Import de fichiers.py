import pandas as pd
import matplotlib as plt
import numpy as np    



def read(file_name):
    df = pd.read_csv(file_name, sep="\t", skiprows=22, decimal=",")
    df = df.iloc[:, 1:].copy() 

    return df.to_numpy()[:, :2]

print(read("convertisseur_090925_01.lvm"))
