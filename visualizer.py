import numpy as np
import pyvista as pv
from utils.validazione import segmento_sicuro
from utils.pathfinding import bfs_deviazione
from utils.geometria import crea_cerchio


def visualizza_superficie_con_percorso(z, lavorabile, path, config):
    n_rows, n_cols = z.shape
    x_vals = np.arange(n_cols)
    y_vals = np.arange(n_rows)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    grid = pv.StructuredGrid()
    grid.points = np.c_[x_grid.ravel(), y_grid.ravel(), z.ravel()]
    grid.dimensions = [n_cols, n_rows, 1]

    plotter = pv.Plotter()
    plotter.add_mesh(grid, cmap="bone", opacity=0.5)

    raggio = config['diametro_cm'] / 2

    if len(path) >= 2:
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            if segmento_sicuro(p1, p2, lavorabile, raggio):
                segment = pv.lines_from_points(np.array([p1, p2]), close=False)
                plotter.add_mesh(segment, color="blue", line_width=2)
            else:
                dev_path = bfs_deviazione(p1, p2, lavorabile, raggio, z)
                if dev_path and len(dev_path) >= 2:
                    for j in range(len(dev_path) - 1):
                        if segmento_sicuro(dev_path[j], dev_path[j+1], lavorabile, raggio):
                            segment = pv.lines_from_points(np.array([dev_path[j], dev_path[j+1]]), close=False)
                            plotter.add_mesh(segment, color="red", line_width=2)

    for punto in path:
        cerchio = crea_cerchio(punto, raggio)
        plotter.add_lines(cerchio, color="red", width=1)

    plotter.add_axes()
    plotter.set_background("white")
    plotter.show(title="Percorso a serpentina della levigatrice")
