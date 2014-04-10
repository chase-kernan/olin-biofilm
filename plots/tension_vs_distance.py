
from biofilm import util
util.set_h5(util.results_h5_path('final_tension_vs_distance'))

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt

# TENSION VS DISTANCE POWERS
# -----------------------------

def compute():
    an.mass.compute_specs(recompute=True)
    an.convex_density.compute_specs(recompute=True)
    an.mean_cell_height.compute_specs(recompute=True)
    an.heights.compute_specs(recompute=True)
    an.perimeter.compute_specs(recompute=True)
    an.roughness.compute_specs(recompute=True)
compute()

# MASS
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-mass'):
        an.mass.phase_diagram_2d('tension_power', 'distance_power',
                                 num_cells=(40, 40))

# CONVEX_DENSITY
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-convex_density'):
        an.convex_density.phase_diagram_2d('tension_power', 'distance_power',
                                           num_cells=(40, 40))

# CELL HEIGHTS
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-cell_height'):
        an.mean_cell_height.phase_diagram_2d('tension_power', 'distance_power',
                                             num_cells=(40, 40))

# HEIGHTS
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-height-mean'):
        an.heights.phase_diagram_2d('tension_power', 'distance_power',
                                    num_cells=(40, 40),
                                    statistic='mean')
    with plot_to_file('tension_vs_distance/tension_vs_distance-height-std'):
        an.heights.phase_diagram_2d('tension_power', 'distance_power',
                                    num_cells=(40, 40),
                                    statistic='std')

# PERIMETER
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-perimeter'):
        an.perimeter.phase_diagram_2d('tension_power', 'distance_power',
                                      num_cells=(40, 40))

# ROUGHNESS
if True:
    with plot_to_file('tension_vs_distance/tension_vs_distance-roughness'):
        an.roughness.phase_diagram_2d('tension_power', 'distance_power',
                                      num_cells=(40, 40))