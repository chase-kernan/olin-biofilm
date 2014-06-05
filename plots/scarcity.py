import numpy as np
import cv2
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, bicg

from biofilm.model import mapping

def make_boundary_layer(cells, width):
    kernel = make_circular_kernel(width)
    boundary_layer = cv2.filter2D(cells.astype(np.uint8), -1, kernel)
    np.logical_not(boundary_layer, out=boundary_layer)

    #remove any non-connected segments
    fill_value = 2
    # (x, y) not (r, c), because fuck you opencv
    fill_source = boundary_layer.shape[1]//2, boundary_layer.shape[0]-1 
    cv2.floodFill(boundary_layer, None, fill_source, fill_value)

    return boundary_layer == fill_value

def make_matrix(cells, boundary, media_ratio):
    n = cells.size
    num_rows, num_cols = cells.shape
    
    center = np.empty(n); center.fill(-4 - media_ratio)
    left = np.ones(n-1)
    right = np.ones_like(left)
    up = np.ones(n-num_cols)
    down = np.ones_like(up)

    # sides
    center[0:num_cols] += 1 # top
    center[num_cols-1::num_cols] += 1 # right
    center[n-num_cols-1:] += 1 # bottom
    center[0::num_cols] += 1 # left
    
    left[num_cols-1::num_cols] = 0
    right[num_cols-1::num_cols] = 0
    
    # outside boundary
    center[boundary] = 1
    left[boundary[1:]] = 0
    right[boundary[:-1]] = 0
    up[boundary[num_cols:]] = 0
    down[boundary[:-num_cols]] = 0
    
    return diags([up, left, center, right, down], 
                 [-num_cols, -1, 0, 1, num_cols], 
                 format='csr')

def do_diffusion(cells, boundary, media_ratio):
    boundary = boundary.flatten()
    M = make_matrix(cells, boundary, media_ratio)
    rhs = np.zeros(cells.size)
    rhs[boundary] = 1
    return spsolve(M, rhs, permc_spec='COLAMD').reshape(cells.shape)

def narrow_cells(model):
    height = model.max_height + model.spec.boundary_layer + 2
    return model.cells[:height, :] > 0

def make_circular_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius, 2*radius))

from matplotlib.pyplot import *


R = 4
mus = np.linspace(0.1, 5.0)
n0 = np.empty_like(mus, dtype=float)

for i, mu in enumerate(mus):
    cells = np.zeros((21, 21), dtype=int)
    cells[10, 10] = 1

    boundary = make_boundary_layer(cells, R)
    conc = do_diffusion(cells, boundary, mu)
    n0[i] = conc[10, 10]

print mus
print n0

plot(mus, n0); hold(True)
plot(mus, 1/(mus*R**2))
show()

Rs = np.arange(1, 9)
mu = 2.0
n0 = np.empty_like(Rs, dtype=float)

for i, R in enumerate(Rs):
    cells = np.zeros((21, 21), dtype=int)
    cells[10, 10] = 1

    boundary = make_boundary_layer(cells, R)
    conc = do_diffusion(cells, boundary, mu)
    n0[i] = conc[10, 10]

print mus
print n0

plot(Rs, n0); hold(True)
plot(Rs, 1/(mu*Rs**2))
show()