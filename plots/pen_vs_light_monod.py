
from biofilm import util
util.set_h5(util.results_h5_path('pen_vs_light_monod'))

from biofilm.model import analysis as an
from biofilm.model import spec as sp
from biofilm.model import runner, result
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt
import numpy as np

import pickle


# LIGHT
# -----

def compute():
    an.mass.compute_specs(recompute=True)
    #an.convex_density.compute_specs(recompute=True)
    an.mean_cell_height.compute_specs(recompute=True)
    # an.heights.compute_specs(recompute=True)
    an.perimeter.compute_specs(recompute=True)
    # an.roughness.compute_specs(recompute=True)
    # an.mean_overhang.compute_specs(recompute=True)
    an.horizontal_surface_area.compute_specs()
# compute()


def get_light_indep_baseline(n=10, fields=['mass', 'convex_density', 'perimeter', 'horizontal_surface_area']):
    util.set_h5(util.results_h5_path('pen_vs_light_monod_base'))

    results = []
    for _ in range(n):
        spec = sp.Spec(stop_on_mass=4000, stop_on_time=50000, stop_on_no_growth=1000,
                       boundary_layer=5, light_penetration=0, tension_power=1, 
                       distance_power=2, media_ratio=1, media_monod=0.25)
        spec.save()
        res = result.from_model(runner.run(spec))
        res.save()
        results.append(res)

    means = {}
    stds = {}
    for field in fields:
        func = getattr(an, field).func
        values = np.array([func(res) for res in results])
        means[field] = values.mean()
        stds[field] = values.std()

    return means, std

if False:
    means = get_light_indep_baseline()
    with open('plots/pen_vs_light_monod-means.cache', 'wb') as f:
        pickle.dump(means, f)

with open('plots/pen_vs_light_monod-means.cache', 'rb') as f:
    baseline_means = pickle.load(f)

print baseline_means

query = 'light_penetration > 2' # '(light_penetration >= 0.1) & (light_penetration <= 16)'

# MASS
if True:
    # with plot_to_file('pen_vs_light_monod/mass'):
    #     an.mass.phase_diagram_2d('light_penetration', 'light_monod',
    #                              num_cells=(75, 50), spec_query=query)
    # with plot_to_file('pen_vs_light_monod/mean_cell_height'):
    #     an.mean_cell_height.phase_diagram_2d('light_penetration', 'light_monod',
    #                              num_cells=(75, 50), spec_query=query)
    with plot_to_file('pen_vs_light_monod/perimeter'):
        an.perimeter.phase_diagram_2d('light_penetration', 'light_monod',
                                 num_cells=(30, 30), spec_query=query)
    with plot_to_file('pen_vs_light_monod/hsa'):
        an.horizontal_surface_area.phase_diagram_2d('light_penetration', 'light_monod',
                                 num_cells=(30, 30), spec_query=query)


