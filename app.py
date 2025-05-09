import pandas as pd
import numpy as np
import pyvista as pv

# Carica la matrice di distanze dal CSV
df = pd.read_csv("surface_merged_at_top.csv", header=None)

# Converti la matrice in array NumPy
distance_matrix = df.to_numpy()

# Crea una griglia di coordinate X, Y
x_vals, y_vals = np.meshgrid(np.arange(distance_matrix.shape[1]), np.arange(distance_matrix.shape[0]))

# Pianifica le coordinate X, Y, Z
x_coords = x_vals.flatten()
y_coords = y_vals.flatten()
z_coords = distance_matrix.flatten()  # La distanza è lungo l'asse Z

# Combina le coordinate in un array di punti (x, y, z)
points = np.column_stack((x_coords, y_coords, z_coords))

# Crea la nuvola di punti in PyVista
cloud = pv.PolyData(points)

# Visualizzazione 3D della nuvola di punti
plotter = pv.Plotter()
plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=5, color="deepskyblue")
plotter.add_axes()
plotter.set_background("white")
plotter.show(title="Visualizzazione scansione 3D dalla matrice")
# Visualizzazione 3D della nuvola di punti
# plotter = pv.Plotter()

plotter.add_points(cloud, point_size=5, color="deepskyblue")
plotter.add_axes()
plotter.set_background("white")
plotter.show(title="Visualizzazione scansione 3D dalla matrice")
