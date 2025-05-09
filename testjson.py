import json
import numpy as np
import pyvista as pv
import pandas as pd
from function import calcola_normale

# Carica superficie
df = pd.read_csv("surface.csv", header=None)
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
z = df.to_numpy()

# Carica il JSON con le coordinate
with open("coordinate_cobot.json", "r") as f:
    data = json.load(f)

points = np.array([[float(pt["x"]), float(pt["y"]), float(pt["z"])] for pt in data["coordinate"]])

# Calcola le normali
normals = np.array([calcola_normale(tuple(p), z) for p in points])

# Linea del percorso
line = pv.lines_from_points(points, close=False)

# Visualizzazione
plotter = pv.Plotter()
plotter.add_mesh(line, color="blue", line_width=2, label="Percorso utensile")

# Aggiungi normali come frecce (facoltativo)
for i in range(0, len(points), 10):  # ogni 10 per meno frecce
    start = points[i]
    direction = normals[i]
    arrow = pv.Arrow(start=start, direction=direction, scale=5)
    plotter.add_mesh(arrow, color="red")

plotter.add_axes()
plotter.set_background("white")
plotter.show(title="Tracciato utensile e inclinazione")
