
from biofilm import util
util.set_h5(util.results_h5_path('final_light'))

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt


# LIGHT
# -----

def compute():
    # an.mass.compute_specs(recompute=True)
    # an.convex_density.compute_specs(recompute=True)
    # an.mean_cell_height.compute_specs(recompute=True)
    # an.heights.compute_specs(recompute=True)
    # an.perimeter.compute_specs(recompute=True)
    # an.roughness.compute_specs(recompute=True)
    # an.mean_overhang.compute_specs(recompute=True)
    an.horizontal_surface_area.compute_specs()
compute()

query = '(light_penetration >= 0.1) & (light_penetration <= 16)'

# MASS
if True:
    with plot_to_file('light/penetration_vs_monod-mass'):
        an.mass.phase_diagram_2d('light_penetration', 'light_monod',
                                 num_cells=(50, 20), spec_query=query)

# CONVEX_DENSITY
if True:
    with plot_to_file('light/penetration_vs_monod-convex_density'):
        an.convex_density.phase_diagram_2d('light_penetration', 'light_monod',
                                           num_cells=(50, 20), spec_query=query)

# CELL HEIGHTS
if True:
    with plot_to_file('light/penetration_vs_monod-cell_height'):
        an.mean_cell_height.phase_diagram_2d('light_penetration', 'light_monod',
                                             num_cells=(50, 20), spec_query=query)

# HEIGHTS
if True:
    with plot_to_file('light/penetration_vs_monod-height-mean'):
        an.heights.phase_diagram_2d('light_penetration', 'light_monod',
                                    num_cells=(50, 20), spec_query=query,
                                    statistic='mean')
    with plot_to_file('light/penetration_vs_monod-height-std'):
        an.heights.phase_diagram_2d('light_penetration', 'light_monod',
                                    num_cells=(50, 20), spec_query=query,
                                    statistic='std')

# PERIMETER
if True:
    with plot_to_file('light/penetration_vs_monod-perimeter'):
        an.perimeter.phase_diagram_2d('light_penetration', 'light_monod',
                                      num_cells=(50, 20), spec_query=query)

# ROUGHNESS
if True:
    with plot_to_file('light/penetration_vs_monod-roughness'):
        an.roughness.phase_diagram_2d('light_penetration', 'light_monod',
                                      num_cells=(50, 20), spec_query=query)

# OVERHANG
if True:
    with plot_to_file('light/penetration_vs_monod-overhang'):
        an.mean_overhang.phase_diagram_2d('light_penetration', 'light_monod',
                                          num_cells=(50, 20), spec_query=query)

# HORIZONTAL SA
if True:
    with plot_to_file('light/penetration_vs_monod-horizontal_sa'):
        an.horizontal_surface_area.phase_diagram_2d('light_penetration', 'light_monod',
                                                    num_cells=(50, 20), spec_query=query)
