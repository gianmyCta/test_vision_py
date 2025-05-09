import numpy as np

def cerchio_valido(row, col, lavorabile, raggio):
    rr, cc = np.ogrid[-raggio:raggio+1, -raggio:raggio+1]
    mask = rr**2 + cc**2 <= raggio**2
    r_start, r_end = int(row - raggio), int(row + raggio + 1)
    c_start, c_end = int(col - raggio), int(col + raggio + 1)
    sub = lavorabile[r_start:r_end, c_start:c_end]
    if sub.shape != mask.shape:
        return False
    return np.all(sub[mask])

def trova_vicino_valido(row, col, lavorabile, raggio):
    for offset in range(1, 6):
        for direction in [-1, 1]:
            new_col = col + offset * direction
            if 0 <= new_col < lavorabile.shape[1]:
                if cerchio_valido(row, new_col, lavorabile, raggio):
                    return new_col
    return None

def segmento_sicuro(p1, p2, lavorabile, raggio):
    x1, y1, _ = p1
    x2, y2, _ = p2
    num = int(np.hypot(x2 - x1, y2 - y1))
    for i in range(num):
        t = i / max(num - 1, 1)
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if not cerchio_valido(y, x, lavorabile, raggio):
            return False
    return True