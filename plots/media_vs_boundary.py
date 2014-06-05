
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
    #an.roughness.compute_specs(recompute=True)
    an.horizontal_surface_area.compute_specs(recompute=True)
    #an.b2r.compute_specs()
    #an.br.compute_specs()
    #an.growth_time.compute_specs(recompute=True)
    #an.recip_b2r.compute_specs()
    pass
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
                                           num_cells=(100, 10), 
                                           vmin=0.65, vmax=1.0)

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

# HORIZONTAL SA
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-horizontal_sa'):
        an.horizontal_surface_area.phase_diagram_2d('media_ratio', 'boundary_layer',
                                                    num_cells=(100, 10))

# B2R MEAN CELL HEIGHT
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-b2r_vs_cell_height'):
        an.b2r.scatter_plot(an.mean_cell_height)
        plt.xlim([0, 30])
        plt.ylim([4, 12])
    with plot_to_file('media_vs_boundary/media_vs_boundary-b2r_vs_convex_density'):
        an.b2r.scatter_plot(an.convex_density)
        plt.xlim([0, 30])
        plt.ylim([0.65, 1])
    with plot_to_file('media_vs_boundary/media_vs_boundary-b2r_vs_perimeter'):
        an.b2r.scatter_plot(an.perimeter)
        plt.xlim([0, 30])
        plt.ylim([2, 6])
    with plot_to_file('media_vs_boundary/media_vs_boundary-b2r_vs_mass'):
        an.b2r.scatter_plot(an.mass)
    with plot_to_file('media_vs_boundary/media_vs_boundary-b2r_vs_growth_time'):
        an.b2r.scatter_plot(an.growth_time)

# RECIP B2R MEAN CELL HEIGHT
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-recip_b2r_vs_cell_height'):
        an.recip_b2r.scatter_plot(an.mean_cell_height)
        plt.xlim([0, 2])
        plt.ylim([4, 12])
    with plot_to_file('media_vs_boundary/media_vs_boundary-recip_b2r_vs_convex_density'):
        an.recip_b2r.scatter_plot(an.convex_density)
        plt.xlim([0, 2])
        plt.ylim([0.65, 1])
    with plot_to_file('media_vs_boundary/media_vs_boundary-recip_b2r_vs_perimeter'):
        an.recip_b2r.scatter_plot(an.perimeter)
        plt.xlim([0, 2])
        plt.ylim([2, 6])

# BR MEAN CELL HEIGHT
if True:
    with plot_to_file('media_vs_boundary/media_vs_boundary-br_vs_cell_height'):
        an.br.scatter_plot(an.mean_cell_height)
    with plot_to_file('media_vs_boundary/media_vs_boundary-br_vs_perimeter'):
        an.br.scatter_plot(an.perimeter)