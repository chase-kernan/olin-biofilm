
import numpy as np
from biofilm.model import mapping

def calculate(model):
    if mode.spec.light_penetration != 0.0:
        np.cumsum(model.cells, axis=0, out=model.light)
        model.light /= -float(model.spec.light_penetration) # otherwise uint16
        np.exp(model.light, out=model.light)
    else:
        model.light.fill(1.0)

def probability(model):
    light = calculate(model)
    return mapping.monod_of_array(light, model.spec.light_monod)