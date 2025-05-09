import pandas as pd
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_gradient_magnitude
import logging
from matplotlib.colors import ListedColormap
from function import cerchio_valido, trova_vicino_valido
from function import esporta_per_cobot
# Configura il logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Parametri utensile
diametro_cm = 30
raggio_cm = diametro_cm / 2
step_cm = 19  # Passo del percorso in cm
pendenza_max = 0.5  # Pendenza massima accettabile

# Carica superficie
logger.info("Caricamento della superficie da 'surface.csv'")
df = pd.read_csv("surface.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
z = df.to_numpy()

n_rows, n_cols = z.shape
logger.info(f"Superficie caricata: {n_rows} righe, {n_cols} colonne")

# Maschera pendenza
gradient = gaussian_gradient_magnitude(z, sigma=1)
lavorabile = gradient < pendenza_max

# Margine di sicurezza (per rimanere nel bordo interno)
margine = int(np.ceil(raggio_cm))

# Costruzione percorso serpentina orizzontale
logger.info("Inizio costruzione del percorso a serpentina orizzontale")
path = []
for col in range(margine, n_cols - margine, step_cm):
    if (col - margine) // step_cm % 2 == 0:
        row_range = range(margine, n_rows - margine, step_cm)
    else:
        row_range = reversed(range(margine, n_rows - margine, step_cm))

    for row in row_range:
        if cerchio_valido(row, col, lavorabile, raggio_cm):
            z_val = z[row, col]
            path.append((col, row, z_val))
        else:
            nuovo_row = trova_vicino_valido(row, col, lavorabile, raggio_cm)
            if nuovo_row is not None:
                z_val = z[nuovo_row, col]
                path.append((col, nuovo_row, z_val))

logger.info(f"Percorso costruito con {len(path)} punti")

# Crea matrice binaria per il percorso (1 se toccato dall'utensile, 0 altrimenti)
lavorato = np.zeros_like(z, dtype=np.uint8)

raggio_cm_int = int(raggio_cm)  # Converti raggio_cm in intero

for punto in path:
    row, col = int(punto[1]), int(punto[0])
    for i in range(-raggio_cm_int, raggio_cm_int + 1):  # Usa raggio_cm_int qui
        for j in range(-raggio_cm_int, raggio_cm_int + 1):  # Usa raggio_cm_int qui
            rr, cc = row + i, col + j
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                if (i**2 + j**2) <= raggio_cm_int**2:
                    lavorato[rr, cc] = 1

# Aggiungi array "lavorazione" alla mesh
grid = pv.StructuredGrid()
x_vals = np.arange(n_cols)
y_vals = np.arange(n_rows)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
grid.dimensions = [n_cols, n_rows, 1]
grid["lavorazione"] = lavorato.ravel(order="C")  # Aggiungi l'array "lavorazione"

# Definisci la mappa di colori: arancione per non toccato, blu per toccato
cmap = ListedColormap(["orange", "blue"])

# Visualizza la superficie e applica il colore
plotter = pv.Plotter()
plotter.add_mesh(
    grid,
    scalars="lavorazione",
    cmap=cmap,
    opacity=0.7,
    show_scalar_bar=False  # Rimuovi la barra dei valori se non necessaria
)

# Aggiungi gli assi e il background bianco
plotter.add_axes()
plotter.set_background("white")
plotter.show(title="Visualizzazione Percorso Utensile")

logger.info("Visualizzazione completata")
