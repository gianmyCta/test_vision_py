import numpy as np
from utils.validazione import cerchio_valido, trova_vicino_valido

def genera_percorso(z, lavorabile, config):
    raggio = config['diametro_cm'] / 2
    step = config['step_cm']
    n_rows, n_cols = z.shape
    margine = int(np.ceil(raggio))

    path = []
    for row in range(margine, n_rows - margine, step):
        if (row - margine) // step % 2 == 0:
            col_range = range(margine, n_cols - margine, step)
        else:
            col_range = reversed(range(margine, n_cols - margine, step))

        for col in col_range:
            if cerchio_valido(row, col, lavorabile, raggio):
                z_val = z[row, col]
                path.append((col, row, z_val))
            else:
                nuovo_col = trova_vicino_valido(row, col, lavorabile, raggio)
                if nuovo_col is not None:
                    z_val = z[row, nuovo_col]
                    path.append((nuovo_col, row, z_val))

    return path