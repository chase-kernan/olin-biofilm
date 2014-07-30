
from biofilm import util
util.set_h5(util.results_h5_path('pen_vs_light_monod_7-27'))

from biofilm.model import analysis as an
from biofilm.model import spec as sp
from biofilm.model import runner, result
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

import pickle
import sys

command = sys.argv[1]

if command == 'dump':
    num_specs = sp.Spec.table.raw.nrows
    print num_specs
    dtype = [('pen', float),
             ('monod', float),
             ('mass', float),
             ('perimeter', float),
             ('hsa', float),
             ('convex_density', float),
             ('cell_height', float)]
    data = np.empty(num_specs, dtype)

    for i, spec in enumerate(sp.Spec.all()):
        data[i]['pen'] = spec.light_penetration
        data[i]['monod'] = spec.light_monod
        data[i]['mass'] = an.mass.get_by_spec(spec)['mean']
        data[i]['perimeter'] = an.perimeter.get_by_spec(spec)['mean']
        data[i]['hsa'] = an.horizontal_surface_area.get_by_spec(spec)['mean']
        data[i]['convex_density'] = an.convex_density.get_by_spec(spec)['mean']
        data[i]['cell_height'] = an.mean_cell_height.get_by_spec(spec)['mean']
        print i, 

    np.save('dump/pen_vs_light_monod', data)

if command == 'baseline':
    def get_light_indep_baseline(n=50, fields=['mass', 'convex_density', 'perimeter', 'horizontal_surface_area']):
        util.set_h5(util.results_h5_path('pen_vs_light_monod_base'))

        results = []
        for _ in range(n):
            spec = sp.Spec(stop_on_mass=4000, stop_on_time=60000, stop_on_no_growth=1000,
                           boundary_layer=7, light_penetration=0, tension_power=1, 
                           distance_power=2, media_ratio=0.51, media_monod=0.25,
                           initial_cell_spacing=2)
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

        return means, stds

    means = get_light_indep_baseline()
    with open('plots/pen_vs_light_monod-means.cache', 'wb') as f:
        pickle.dump(means, f)

if command == 'plot':
    data = np.load('dump/pen_vs_light_monod.npy')

    with open('plots/pen_vs_light_monod-means.cache', 'rb') as f:
        baseline_means, baseline_stds = pickle.load(f)

    print baseline_means, baseline_stds

    query = 'light_penetration > 2' # '(light_penetration >= 0.1) & (light_penetration <= 16)'
    
    def plot_phase(field, num_cells=(64, 32), **plot_args):
        xs = np.reshape(data['pen'], (len(data), 1))
        ys = np.reshape(data['monod'], (len(data), 1))
        values = np.reshape(data[field], (len(data), 1))

        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()
        
        assert xMin != xMax
        assert yMin != yMax

        try:
            num_x, num_y = num_cells
        except TypeError:
            num_x, num_y = num_cells, num_cells
        
        grid = np.mgrid[xMin:xMax:num_x*1j, 
                        yMin:yMax:num_y*1j]
        interp = interpolate.griddata(np.hstack((xs, ys)), 
                                      values, 
                                      np.vstack((grid[0].flat, grid[1].flat)).T, 
                                      'linear')
        valueGrid = np.reshape(interp, grid[0].shape)
        
        plt.pcolormesh(grid[0], grid[1], valueGrid, **plot_args)
        plt.xlim(xMin, xMax)
        plt.ylim(yMin, yMax)
        plt.colorbar()

    with plot_to_file('pen_vs_light_monod/perimeter'):
        plot_phase('perimeter')
    with plot_to_file('pen_vs_light_monod/mass'):
        plot_phase('mass')
    with plot_to_file('pen_vs_light_monod/hsa'):
        plot_phase('hsa')
    with plot_to_file('pen_vs_light_monod/convex_density'):
        plot_phase('convex_density')


