
import numpy as np

def calculate(biofilm_image):
    return biofilm_image.sum(axis=1)/float(biofilm_image.shape[1])