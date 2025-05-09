import json
import numpy as np
import pyvista as pv

# Carica il file JSON
with open("superficie_utensile.json", "r") as f:
    data = json.load(f)

# Estrai i dati della superficie
superficie = data["superficie"]
dimensioni = superficie["dimensioni"]
z_data = np.array(superficie["dati"])

# Estrai il percorso dell'utensile
percorso = data["percorso_utensile"]["dati"]
percorso_points = np.array([[p["x"], p["y"], p["z"]] for p in percorso])

# Crea la griglia per la superficie
n_cols, n_rows = dimensioni
x_vals = np.arange(n_cols)
y_vals = np.arange(n_rows)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)

# Crea il StructuredGrid per la superficie
grid = pv.StructuredGrid()
grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z_data.ravel()]
grid.dimensions = [n_cols, n_rows, 1]

# Assicurati che il percorso dell'utensile non superi la superficie
for i, point in enumerate(percorso_points):
    x, y, z = point
    if z > z_data[int(y), int(x)]:  # Controlla se il punto supera la superficie
        print(f"Il punto ({x}, {y}) supera la superficie. Limito la z a {z_data[int(y), int(x)]}")
        percorso_points[i, 2] = z_data[int(y), int(x)]  # Limita il valore z al massimo valore della superficie

# Crea un plotter
plotter = pv.Plotter()

# Aggiungi la superficie 3D
plotter.add_mesh(grid, cmap="viridis", opacity=0.7, show_edges=True, label="Superficie")

# Aggiungi il percorso dell'utensile
plotter.add_points(percorso_points, color="red", point_size=10, label="Percorso Utensile")

# Crea una lista di segmenti, con i punti consecutivi per formare le linee
segments = []
for i in range(len(percorso_points) - 1):
    segments.append([percorso_points[i], percorso_points[i + 1]])

# Converti i segmenti in una lista di punti da passare a add_lines
segments_points = []
for segment in segments:
    segments_points.append(segment[0])  # Punti iniziali
    segments_points.append(segment[1])  # Punti finali

# Converti la lista in un array NumPy
segments_points = np.array(segments_points)

# Aggiungi le linee al plotter
plotter.add_lines(segments_points, color="blue", width=2)

# Aggiungi assi e altre configurazioni
plotter.add_axes()
plotter.set_background("white")
plotter.show(title="Visualizzazione Percorso Utensile su Superficie")
