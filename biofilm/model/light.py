
import numpy as np
from biofilm.model import mapping

def calculate(model):
    if model.spec.light_penetration > 0:
        light = np.cumsum(model.cells[:model.max_height+1, :], 
                          axis=0, dtype=float)
        light /= -float(model.spec.light_penetration) # otherwise uint16
        model.light = np.exp(light)
    else:
        model.light = np.ones((model.max_height, model.cells.shape[1]))
    return model.light

def probability(model):
    light = calculate(model)
    if model.spec.light_penetration > 0:
        light = mapping.monod_of_array(light, model.spec.light_monod)
    return light