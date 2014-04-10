
from biofilm import util
util.set_h5(util.results_h5_path('final_boundary_vs_media_ratio'))

# MEDIA_RATIO VS BOUNDARY LAYER
# -----------------------------

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt

# TENSION VS DISTANCE POWERS
# -----------------------------

def compute():
    #an.mass.compute_specs(recompute=True)
    #an.convex_density.compute_specs(recompute=True)
    #an.mean_cell_height.compute_specs(recompute=True)
    #an.heights.compute_specs(recompute=True)
    #an.perimeter.compute_specs(recompute=True)
    an.roughness.compute_specs(recompute=True)
compute()

# MASS
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-mass'):
        an.mass.phase_diagram_2d('media_ratio', 'boundary_layer',
                                 num_cells=(100, 10))

# CONVEX_DENSITY
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-convex_density'):
        an.convex_density.phase_diagram_2d('media_ratio', 'boundary_layer',
                                           num_cells=(100, 10))

# CELL HEIGHTS
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-cell_height'):
        an.mean_cell_height.phase_diagram_2d('media_ratio', 'boundary_layer',
                                             num_cells=(100, 10))

# HEIGHTS
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-height-mean'):
        an.heights.phase_diagram_2d('media_ratio', 'boundary_layer',
                                    num_cells=(100, 10),
                                    statistic='mean')
    with plot_to_file('media_vs_boundary/media_vs_boundary-height-std'):
        an.heights.phase_diagram_2d('media_ratio', 'boundary_layer',
                                    num_cells=(100, 10),
                                    statistic='std')

# PERIMETER
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-perimeter'):
        an.perimeter.phase_diagram_2d('media_ratio', 'boundary_layer',
                                      num_cells=(100, 10))

# ROUGHNESS
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-roughness'):
        an.roughness.phase_diagram_2d('media_ratio', 'boundary_layer',
                                      num_cells=(100, 10))