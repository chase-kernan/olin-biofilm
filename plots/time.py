
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

light_dep = '(light_penetration == 0)'
light_indep = '(light_penetration >= 4)'

# MASS
if True:
    with plot_to_file('time/spacing-mass'):
        an.mass.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Dependent', color='b', marker='o')
        an.mass.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Independent')
        plt.legend()

# CONVEX_DENSITY
if True:
    with plot_to_file('time/spacing-convex_density'):
        an.convex_density.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Dependent')
        an.convex_density.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Independent')
        plt.legend()

# CELL HEIGHTS
if True:
    with plot_to_file('time/spacing-cell_height'):
        an.mean_cell_height.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Dependent')
        an.mean_cell_height.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Independent')
        plt.legend()

# HEIGHTS
if True:
    with plot_to_file('time/spacing-height-mean'):
        an.heights.plot('initial_cell_spacing', statistic='mean',
                     spec_query=light_dep, label='Light Dependent')
        an.heights.plot('initial_cell_spacing', statistic='mean',
                     spec_query=light_dep, label='Light Independent')
        plt.legend()
    with plot_to_file('time/spacing-height-std'):
        an.heights.plot('initial_cell_spacing', statistic='std',
                     spec_query=light_dep, label='Light Dependent')
        an.heights.plot('initial_cell_spacing', statistic='std',
                     spec_query=light_dep, label='Light Independent')
        plt.legend()

# PERIMETER
if True:
    with plot_to_file('time/spacing-perimeter'):
        an.perimeter.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Dependent')
        an.perimeter.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Independent')
        plt.legend()

# ROUGHNESS
if True:
    with plot_to_file('time/spacing-roughness'):
        an.roughness.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Dependent')
        an.roughness.plot('initial_cell_spacing', 
                     spec_query=light_dep, label='Light Independent')
        plt.legend()