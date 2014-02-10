
from biofilm.classify import flat
from biofilm.model import media
from solver_diagnostics import solver_diagnostics
import numpy as np

cells = flat.make_image()
boundary = media.make_boundary_layer(cells, 5).flatten()
M = media.make_matrix(cells, boundary, 1.0)
rhs = np.zeros(cells.size)
rhs[boundary] = 1

from pyamg.classical.classical import ruge_stuben_solver
import scipy
from convergence_tools import print_cycle_history

ml = ruge_stuben_solver(M, max_coarse=500)
x0 = np.zeros_like(rhs)#np.linspace(0, 1, cells.shape[0])[:, np.newaxis].repeat(cells.shape[1], axis=1)

resvec = []
x = ml.solve(rhs, x0=x0, maxiter=20, tol=1e-8, residuals=resvec)
print_cycle_history(resvec, ml, verbose=True, plotting=True)

#solver_diagnostics(M)

#from solver_diagnostic import solver_diagnostic
#solver_diagnostic(M)