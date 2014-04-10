
from biofilm import util
util.set_h5(util.results_h5_path('final_media'))

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt

# MEDIA_RATIO VS BOUNDARY LAYER
# -----------------------------

def compute():
    #an.mass.compute_specs()
    #an.convex_density.compute_specs()
    #an.mean_cell_height.compute_specs()
    #an.heights.compute_specs()
    #an.perimeter.compute_specs()
    an.roughness.compute_specs(recompute=True)
    pass
compute()

# MASS
if False:
    with plot_to_file('media/ratio_vs_monod-mass'):
        an.mass.phase_diagram_2d('media_ratio', 'media_monod', 
                                       num_cells=(40, 40))

# CONVEX_DENSITY
if False:
    with plot_to_file('media/ratio_vs_monod-convex_density'):
        an.convex_density.phase_diagram_2d('media_ratio', 'media_monod',
                                           num_cells=(40, 40))

# CELL HEIGHTS
if False:
    with plot_to_file('media/ratio_vs_monod-cell_height'):
        an.mean_cell_height.phase_diagram_2d('media_ratio', 'media_monod',
                                             num_cells=(40, 40))

# HEIGHTS
if False:
    with plot_to_file('media/ratio_vs_monod-height-mean'):
        an.heights.phase_diagram_2d('media_ratio', 'media_monod',
                                    num_cells=(40, 40),
                                    statistic='mean')
    with plot_to_file('media/ratio_vs_monod-height-std'):
        an.heights.phase_diagram_2d('media_ratio', 'media_monod',
                                    num_cells=(40, 40),
                                    statistic='std')

# PERIMETER
if False:
    with plot_to_file('media/ratio_vs_monod-perimeter'):
        an.perimeter.phase_diagram_2d('media_ratio', 'media_monod',
                                      num_cells=(40, 40))

# ROUGHNESS
if True:
    with plot_to_file('media/ratio_vs_monod-roughness'):
        an.roughness.phase_diagram_2d('media_ratio', 'media_monod',
                                      num_cells=(40, 40))