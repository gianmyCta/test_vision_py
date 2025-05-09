from venv import logger
import pandas as pd
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_gradient_magnitude
import matplotlib.pyplot as plt
from collections import deque
import json
import logging

# Configura il logger
logger = logging.getLogger(__name__)


file_list = ["surface_top.csv", "surface_bot.csv"]

def pulisci_csv(file_list):
    """
    Pulisce ciascun file CSV nella lista:
    - Sostituisce celle vuote o NaN con 0
    - Rende la matrice regolare (stesso numero di colonne per riga)
    - Salva un nuovo file con suffisso '_pulito.csv'
    - Esegue un merge verticale e salva in 'merge_superfici.csv'
    """
    dfs_puliti = []

    for file_name in file_list:
        # Carica il CSV iniziale (tratta spazi vuoti come NaN)
        df_temp = pd.read_csv(file_name, na_values=["", " "], header=None)

        # Trova il numero massimo di colonne
        max_cols = df_temp.shape[1]

        # Ricarica con nomi fissi di colonna (per righe irregolari)
        df = pd.read_csv(file_name, na_values=["", " "], header=None, names=range(max_cols))

        # Riempie i NaN con 0
        df.fillna(0, inplace=True)

        # Salva con suffisso _pulito
        output_file = file_name.replace(".csv", "_pulito.csv")
        df.to_csv(output_file, index=False, header=False)

        print(f"{file_name} → trasformato in matrice regolare e salvato come {output_file}")

        # Aggiungi ai dati da unire
        dfs_puliti.append(df)

    # Esegui il merge verticale
    merged_df = pd.concat(dfs_puliti, ignore_index=True, axis=1)
    merged_df.to_csv("merge_superfici.csv", index=False, header=False)
    print("✔ Merge verticale completato: salvato in 'merge_superfici.csv'")


    """
    Verifica che l'intera area del cerchio centrato in (x, y) sia lavorabile.
    """
    r = int(np.ceil(raggio))
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx**2 + dy**2 <= r**2:
                yy, xx = y + dy, x + dx
                if not (0 <= yy < lavorabile.shape[0] and 0 <= xx < lavorabile.shape[1]):
                    return False  # Fuori dai bordi
                if not lavorabile[yy, xx]:
                    return False  # Ostacolo
    return True

def crea_cerchio_con_normale(punto, raggio_cm, z, n_punti=36):
    """
    Crea un cerchio 3D sul piano locale tangente alla superficie nel punto dato.

    Args:
        punto (tuple): Coordinate (x, y, z)
        raggio_cm (float): Raggio del cerchio
        z (2D array): Superficie
        n_punti (int): Numero di punti per disegnare il cerchio (maggiore = più preciso)

    Returns:
        np.ndarray: Array (n_punti, 3) con coordinate 3D del cerchio
    """
    x, y, z0 = punto
    x, y = int(round(x)), int(round(y))

    # Calcolo delle derivate parziali per stimare la normale
    dzdx = (z[y, min(x+1, z.shape[1]-1)] - z[y, max(x-1, 0)]) / 2.0
    dzdy = (z[min(y+1, z.shape[0]-1), x] - z[max(y-1, 0), x]) / 2.0
    normale = np.array([-dzdx, -dzdy, 1.0])
    normale /= np.linalg.norm(normale)

    # Due vettori ortogonali al vettore normale (tangenti al piano)
    if abs(normale[2]) > 1e-6:
        v1 = np.cross(normale, [0, 0, 1])
    else:
        v1 = np.cross(normale, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normale, v1)

    # Generazione dei punti del cerchio
    angoli = np.linspace(0, 2*np.pi, n_punti, endpoint=False)
    cerchio = [punto + raggio_cm * (np.cos(a) * v1 + np.sin(a) * v2) for a in angoli]
    return np.array(cerchio)


def cerchio_valido(y, x, lavorabile, raggio):
    r = int(np.ceil(raggio))
    
    # Definisci il bounding box
    y_start, y_end = y - r, y + r + 1
    x_start, x_end = x - r, x + r + 1

    # Controlla che sia tutto nel range valido
    if y_start < 0 or y_end > lavorabile.shape[0] or x_start < 0 or x_end > lavorabile.shape[1]:
        return False

    # Crea maschera circolare
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    mask = (xx**2 + yy**2) <= r**2

    # Estrai la regione da controllare
    regione = lavorabile[y_start:y_end, x_start:x_end]

    # Applica la maschera circolare e verifica che tutti i punti siano validi
    return np.all(regione[mask])

def trova_vicino_valido(riga, colonna_start, lavorabile, raggio, step=1, max_distanza=10):
    """Trova la colonna più vicina valida nella stessa riga (entro max_distanza)."""
    # Definisci l'intervallo di colonne da esplorare
    colonne_da_verificare = np.concatenate([np.arange(colonna_start - d, colonna_start + d + 1, step) 
                                            for d in range(1, max_distanza + 1)])
    
    # Filtra le colonne valide (entro il range e controllando se sono valide con cerchio_valido)
    colonne_validi = colonne_da_verificare[(colonne_da_verificare >= 0) & (colonne_da_verificare < lavorabile.shape[1])]
    
    for col in colonne_validi:
        if cerchio_valido(riga, col, lavorabile, raggio):
            return col
    return None  # Nessun punto valido vicino

def segmento_sicuro(p1, p2, lavorabile, raggio):
    """Verifica che la linea tra due punti non attraversi zone non lavorabili."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    num_steps = int(np.linalg.norm([x2 - x1, y2 - y1])) * 2
    for t in np.linspace(0, 1, num_steps):
        x = int(round(x1 + t * (x2 - x1)))
        y = int(round(y1 + t * (y2 - y1)))
        if not cerchio_valido(y, x, lavorabile, raggio):
            return False
    return True

def crea_percorso_serpentina(riga, colonna_start, lavorabile, raggio, z, step=1):
    """Genera un percorso a serpentina a partire da una riga e colonna iniziali."""
    path = []
    for col in range(colonna_start, lavorabile.shape[1] - raggio, step):
        if cerchio_valido(riga, col, lavorabile, raggio):
            z_val = z[riga, col]
            path.append((col, riga, z_val))
        else:
            nuovo_col = trova_vicino_valido(riga, col, lavorabile, raggio)
            if nuovo_col is not None:
                z_val = z[riga, nuovo_col]
                path.append((nuovo_col, riga, z_val))
    return path

def bfs_deviazione(start, end, lavorabile, raggio, z, max_dev=20):
    """
    Cerca un percorso sicuro da start a end evitando ostacoli, entro un'area limitata.
    Ritorna una lista di punti intermedi (deviazione).
    """
    rows, cols = lavorabile.shape
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add((int(start[1]), int(start[0])))

    dirs = [(-1,0),(1,0),(0,-1),(0,1)]  # solo 4 direzioni (oppure 8 con diagonali)

    while queue:
        current, path = queue.popleft()
        cy, cx = int(current[1]), int(current[0])
        
        if abs(cx - end[0]) <= 1 and abs(cy - end[1]) <= 1:
            return path + [end]
        
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if (ny, nx) not in visited and cerchio_valido(ny, nx, lavorabile, raggio):
                    if abs(ny - start[1]) <= max_dev and abs(nx - start[0]) <= max_dev:
                        visited.add((ny, nx))
                        nuovo_punto = (nx, ny, z[ny, nx])
                        queue.append((nuovo_punto, path + [nuovo_punto]))

    return None  # Nessuna deviazione trovata

def custom_serializer(obj):
    if isinstance(obj, np.int64):
        return int(obj)  # Converte np.int64 in int
    if isinstance(obj, np.float64):
        return float(obj)  # Converte np.float64 in float
    raise TypeError(f"Type {type(obj)} not serializable")

import json
import numpy as np

import json

def esporta_per_cobot(punti, z_matrix, output_path):
    """
    Esporta i punti con normali in formato compatibile con RoboDK.
    
    Args:
        punti (list): Lista di tuple (x, y, z)
        z_matrix (ndarray): Matrice di elevazione
        output_path (str): Percorso file JSON
    """
    from function import calcola_normale  # Se non già importata

    output_data = []

    for punto in punti:
        x, y, z = punto
        nx, ny, nz = calcola_normale((x, y, z), z_matrix)

        output_data.append({
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "nx": float(nx),
            "ny": float(ny),
            "nz": float(nz)
        })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Esportazione completata in {output_path}")



def visualizza_utensile_su_percorso(path, z, raggio_cm, n_rows, n_cols, altezza_utensile=0.2):
    """
    Visualizza il percorso dell'utensile sovrapposto alla superficie, con cilindri orientati per ogni punto del percorso.
    
    Args:
        path (list of tuples): Lista di punti (x, y, z) del percorso
        z (np.array): La matrice della superficie (altezza z in ogni punto)
        raggio_cm (float): Raggio dell'utensile in centimetri
        n_rows (int): Numero di righe della superficie (altezza)
        n_cols (int): Numero di colonne della superficie (larghezza)
        altezza_utensile (float): Altezza dell'utensile (cilindro)
    """
    # Crea la mesh della superficie
    x_vals = np.arange(n_cols)
    y_vals = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]

    # Crea il plotter per la visualizzazione
    plotter = pv.Plotter()
    
    # Aggiungi la superficie al plot
    plotter.add_mesh(grid, cmap="bone", opacity=0.5)
    
    # Aggiungi i cilindri dell'utensile al percorso
    for punto in path:
        x, y, z_val = punto

        # Calcola la normale alla superficie
        if 1 <= y < n_rows-1 and 1 <= x < n_cols-1:
            dzdy = z[int(y+1)] - z[int(y-1)]
            dzdx = z[int(y), int(x+1)] - z[int(y), int(x-1)]
            normal = np.array([-dzdx, -dzdy, 1.0])
            normal = normal / np.linalg.norm(normal)
        else:
            normal = np.array([0, 0, 1])  # normale verticale di default

        # Crea il cilindro orientato
        cilindro = pv.Cylinder(center=(x, y, z_val), direction=normal, radius=raggio_cm, height=altezza_utensile)
        plotter.add_mesh(cilindro, color="orange", opacity=0.4)

    # Aggiungi assi e altre opzioni visive
    plotter.add_axes()
    plotter.set_background("white")
    plotter.show(title="Visualizzazione Percorso Utensile")


def calcola_normale(punto, z_matrix):
    """
    Calcola la normale alla superficie nel punto dato (x, y, z)
    
    Args:
        punto (tuple): Coordinata (x, y, z)
        z_matrix (ndarray): Matrice di elevazione
    
    Returns:
        tuple: Vettore normale (nx, ny, nz) normalizzato
    """
    x, y, _ = punto
    x = int(round(x))
    y = int(round(y))
    
    n_rows, n_cols = z_matrix.shape
    
    # Gestione dei bordi
    if x <= 0 or x >= n_cols - 1 or y <= 0 or y >= n_rows - 1:
        return (0.0, 0.0, 1.0)  # Normale verticale di default ai bordi

    # Derivate finite centrali
    dzdx = (z_matrix[y, x + 1] - z_matrix[y, x - 1]) / 2.0
    dzdy = (z_matrix[y + 1, x] - z_matrix[y - 1, x]) / 2.0

    # Vettore normale (verso z positivo)
    nx = -dzdx
    ny = -dzdy
    nz = 1.0

    # Normalizzazione
    norma = np.sqrt(nx**2 + ny**2 + nz**2)
    return (nx / norma, ny / norma, nz / norma)

def correggi_z(punto, superficie):
    x, y, z = punto
    # Trova l'altezza della superficie per il punto (x, y)
    z_superficie = superficie[int(y), int(x)]
    # Assicurati che la z non sia mai inferiore all'altezza della superficie
    if z < z_superficie:
        z = z_superficie
    return (x, y, z)
