import pandas as pd
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_gradient_magnitude
from function import *
from config import *
import logging
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

# Parametri utensile ##DA MODIFICARE XKE IN GIA ON CONFIG
diametro_cm = 30
raggio_cm = diametro_cm / 2
step_cm = 19 # Passo del percorso in cm
pendenza_max = 0.7 #agire su questo per aumentare la pendenza massima accettabile

# Carica superficie
logger.info("Caricamento della superficie da 'surface.csv'")
df = pd.read_csv("surface.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Assicurati che questa riga venga eseguita prima di chiamare la funzione
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
            logger.debug(f"Punto valido aggiunto: ({col}, {row}, {z_val})")
        else:
            nuovo_row = trova_vicino_valido(row, col, lavorabile, raggio_cm)
            if nuovo_row is not None:
                z_val = z[nuovo_row, col]
                path.append((col, nuovo_row, z_val))
                logger.debug(f"Punto alternativo trovato: ({col}, {nuovo_row}, {z_val})")
            else:
                logger.warning(f"Nessun punto valido trovato vicino a ({row}, {col})")

logger.info(f"Percorso costruito con {len(path)} punti")

logger.info("Inizio visualizzazione della superficie e del percorso")

# Mesh superficie
x_vals = np.arange(n_cols)
y_vals = np.arange(n_rows)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
grid = pv.StructuredGrid()
grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
grid.dimensions = [n_cols, n_rows, 1]

# Visualizzazione
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="bone", opacity=0.5)

# Traccia la linea del percorso serpentina (solo segmenti sicuri)
if len(path) >= 2:
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        if segmento_sicuro(p1, p2, lavorabile, raggio_cm):
            segment = pv.lines_from_points(np.array([p1, p2]), close=False)
            plotter.add_mesh(segment, color="blue", line_width=2)
        else:
            dev_path = bfs_deviazione(p1, p2, lavorabile, raggio_cm, z)
            if dev_path and len(dev_path) >= 2:
                for j in range(len(dev_path) - 1):
                    if segmento_sicuro(dev_path[j], dev_path[j+1], lavorabile, raggio_cm):
                        segment = pv.lines_from_points(np.array([dev_path[j], dev_path[j+1]]), close=False)
                        plotter.add_mesh(segment, color="red", line_width=2)

# Aggiungi i cerchi del passaggio utensile in modo vettoriale (cerchi chiusi)
cerchi = []
for punto in path:
    cerchio = crea_cerchio_con_normale(punto, raggio_cm, z, n_punti=100)  # Risoluzione aumentata
    cerchio = np.vstack([cerchio, cerchio[0]])  # Chiudi il cerchio
    segments = np.array([[cerchio[i], cerchio[i + 1]] for i in range(len(cerchio) - 1)])
    segments = segments.reshape(-1, 3)  # Flatten per PyVista
    cerchi.append(segments)

# Unisci tutti i segmenti in un unico array
tutti_segmenti = np.vstack(cerchi)

# Aggiungi tutte le linee al plotter in un'unica chiamata
plotter.add_lines(tutti_segmenti, color="red", width=1)

plotter.add_axes()
plotter.set_background("white")

plotter.show(title="Percorso a serpentina della levigatrice")
logger.info("Visualizzazione completata")

# Calcola le normali
normals = np.array([calcola_normale(tuple(p), z) for p in path])

# Esporta in JSON con le normali
esporta_per_cobot(path, z, "coordinate_cobot.json")

logger.info("Esportazione completata")
logger.info(f"Numero di punti nel percorso: {len(path)}")
logger.debug(f"Percorso: {path}")
