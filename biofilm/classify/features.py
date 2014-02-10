
import numpy as np

def calculate(biofilm_image):
    coverages = np.zeros(20)
    calculated = biofilm_image.sum(axis=1)/float(biofilm_image.shape[1])
    coverages[:min(calculated.size, 20)] = calculated[:20]
    return coverages 