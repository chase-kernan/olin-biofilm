
from biofilm import util
util.set_h5(util.results_h5_path('final_time'))

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt


# TIME
# -----

def compute():
    an.mass.compute_specs(recompute=True)
    an.convex_density.compute_specs(recompute=True)
    an.mean_cell_height.compute_specs(recompute=True)
    an.heights.compute_specs(recompute=True)
    an.perimeter.compute_specs(recompute=True)
    an.roughness.compute_specs(recompute=True)
#compute()

query = '(light_penetration == 0)'

# MASS
if True:
    with plot_to_file('spacing/mass'):
        an.mass.plot('initial_cell_spacing', spec_query=query)

# CONVEX_DENSITY
if True:
    with plot_to_file('spacing/convex_density'):
        an.convex_density.plot('initial_cell_spacing', 
                               spec_query=query)

# CELL HEIGHTS
if True:
    with plot_to_file('spacing/cell_height'):
        an.mean_cell_height.plot('initial_cell_spacing',
                                 spec_query=query)

# HEIGHTS
if True:
    with plot_to_file('spacing/height-mean'):
        an.heights.plot('initial_cell_spacing',
                        spec_query=query, statistic='mean')
    with plot_to_file('spacing/height-std'):
        an.heights.plot('initial_cell_spacing',
                        spec_query=query, statistic='std')

# PERIMETER
if True:
    with plot_to_file('spacing/perimeter'):
        an.perimeter.plot('initial_cell_spacing',
                          spec_query=query)

# ROUGHNESS
if True:
    with plot_to_file('spacing/roughness'):
        an.roughness.plot('initial_cell_spacing',
                          spec_query=query)