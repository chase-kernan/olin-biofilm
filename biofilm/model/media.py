
import numpy as np
import cv2

from scipy.ndimage.filters import laplace
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from biofilm.model import mapping


def calculate(model):
    cells = narrow_cells(model.cells)
    boundary_layer = make_boundary_layer(cells, model.spec.boundary_layer)
    media = do_diffusion(cells, boundary, model.spec.media_ratio)

def probability(model):
    """NOTE: this does not return an array of the same size as cells.

    It has been shrunk by narrow_cells. Same width, different height."""
    media = calculate(model)
    return mapping.monod_of_array(media, model.spec.media_monod)

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
    
def do_diffusion(cells, boundary, media_ratio):
    n = cells.size
    num_rows, num_cols = cells.shape
    
    center = np.empty(n); center.fill(-4 - media_ratio)
    left = np.ones(n-1)
    right = np.ones_like(left)
    up = np.ones(n-num_cols)
    down = np.ones_like(up)
    
    rhs = np.zeros_like(center)
    
    # sides
    center[0:num_cols] += 1 # top
    center[num_cols-1::num_cols] += 1 # right
    center[n-num_cols-1:] += 1 # bottom
    center[0::num_cols] += 1 # left
    
    left[num_cols-1::num_cols] = 0
    right[num_cols-1::num_cols] = 0
    
    # outside boundary
    boundary = boundary.flatten()
    center[boundary] = 1
    rhs[boundary] = 1
    left[boundary[1:]] = 0
    right[boundary[:-1]] = 0
    up[boundary[num_cols:]] = 0
    down[boundary[:-num_cols]] = 0
    
    M = diags([up, left, center, right, down], 
              [-num_cols, -1, 0, 1, num_cols], 
              format='csr')
    return spsolve(M, rhs).reshape(cells.shape)

def narrow_cells(model):
    height = model.max_height + model.spec.boundary_layer + 2
    return self.cells[:height, :] > 0

def make_circular_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius, 2*radius))