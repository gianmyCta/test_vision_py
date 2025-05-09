import pandas as pd
import numpy as np

def carica_superficie(filepath):
    df = pd.read_csv(filepath, header=None)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df.to_numpy()

def calcola_maschera_lavorabile(gradient, soglia):
    return gradient < soglia