import numpy as np
import pyvista as pv

def crea_cerchio(punto, raggio, num_points=50):
    
    x0, y0, z0 = punto
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = x0 + raggio * np.cos(theta)
    y = y0 + raggio * np.sin(theta)
    z = np.full_like(x, z0)
    points = np.column_stack((x, y, z))
    lines = np.array([[1, i, i+1] for i in range(num_points - 1)] + [[1, num_points - 1, 0]])
    return points