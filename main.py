import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

import config
from preprocessing import carica_superficie, calcola_maschera_lavorabile
from planner import genera_percorso
from visualizer import visualizza_superficie_con_percorso

z = carica_superficie(config.CONFIG['surface_file'])
gradient = gaussian_gradient_magnitude(z, sigma=1)
lavorabile = calcola_maschera_lavorabile(gradient, config.CONFIG['pendenza_max'])
path = genera_percorso(z, lavorabile, config.CONFIG)
visualizza_superficie_con_percorso(z, lavorabile, path, config.CONFIG)