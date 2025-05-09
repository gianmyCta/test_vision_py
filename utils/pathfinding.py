import pandas as pd
import numpy as np
from collections import deque

def bfs_deviazione(start, end, lavorabile, raggio, z):
    from collections import deque
    n_rows, n_cols = lavorabile.shape
    visited = set()
    queue = deque([(start, [start])])

    def vicini(p):
        x, y, _ = p
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= ny < n_rows and 0 <= nx < n_cols:
                if lavorabile[int(ny), int(nx)]:
                    yield (nx, ny, z[int(ny), int(nx)])

    while queue:
        current, path = queue.popleft()
        if np.hypot(current[0] - end[0], current[1] - end[1]) <= raggio:
            return path + [end]
        for neighbor in vicini(current):
            if (neighbor[0], neighbor[1]) not in visited:
                visited.add((neighbor[0], neighbor[1]))
                queue.append((neighbor, path + [neighbor]))
    return []