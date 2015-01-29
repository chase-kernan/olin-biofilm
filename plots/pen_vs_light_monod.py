
from biofilm import util
util.set_h5(util.results_h5_path('pen_vs_light_monod_aug-8'))

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
    specs = list(sp.Spec.where('(stop_on_time > 180000) | (light_penetration >= 5)'))
    num_specs = len(specs)
    print num_specs
    dtype = [('pen', float),
             ('monod', float),
             ('mass', float),
             ('perimeter', float),
             ('hsa', float),
             ('convex_density', float),
             ('cell_height', float)]
    data = np.empty(num_specs, dtype)

    for i, spec in enumerate(specs):
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
    def get_light_indep_baseline(n=50, fields=['mass', 'convex_density', 'perimeter', 'horizontal_surface_area', 'mean_cell_height']):
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
    from matplotlib.gridspec import GridSpec
    from matplotlib import colorbar as cb

    data = np.load('dump/pen_vs_light_monod.npy')

    with open('plots/pen_vs_light_monod-means.cache', 'rb') as f:
        baseline_means, baseline_stds = pickle.load(f)

    field_map = {'hsa': 'horizontal_surface_area', 'cell_height':'mean_cell_height'}
    for field_to, field_from in field_map.items():
        baseline_means[field_to] = baseline_means[field_from]
        baseline_stds[field_to] = baseline_stds[field_from]
    print baseline_means, baseline_stds

    query = 'light_penetration > 2' # '(light_penetration >= 0.1) & (light_penetration <= 16)'
    
    def plot_phase(field, num_cells=(32, 32), tick_values=[], ylabel='', fig_label='A', **plot_args):
        xs = np.reshape(data['pen'], (len(data), 1))
        ys = np.reshape(data['monod'], (len(data), 1))

        values = data[field].copy()
        good_values = (data['mass'] > 3700) & (data['pen'] >= 2.75)
        values[np.logical_not(good_values)] = values[good_values].min() # 0
        values = np.reshape(values, (len(data), 1))

        xMin, xMax = xs.min(), xs.max()
        yMin, yMax = ys.min(), ys.max()

        xMin = 2.75
        
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

        fig = plt.figure(figsize=(6.83/2.05, 3.40))
        gs = GridSpec(100,100, bottom=0.18,left=0.18,right=0.88)
        phase_ax = fig.add_subplot(gs[:,:82])
        color_ax = fig.add_subplot(gs[:,94:])
        
        pcolor = phase_ax.pcolormesh(grid[0], grid[1], valueGrid, **plot_args)
        phase_ax.set_xlim(xMin, xMax)
        phase_ax.set_ylim(yMin, yMax)
        phase_ax.set_xlabel('$p$')
        phase_ax.set_ylabel('$I_{1/2}$')
        phase_ax.tick_params(top='off', right='off', labelsize=8)

        fig.text(0.09, 0.10, fig_label, fontsize=12, fontweight='bold', transform=phase_ax.transAxes)

        v_max = values[values > 0].max()
        v_min = values[values > 0].min()

        gradient = np.outer(np.linspace(0, 1, 100), np.ones(10))
        color_ax.pcolor(gradient, cmap='hot')
        color_ax.tick_params(labelsize=8)
        color_ax.set_xticks([])
        ylim = color_ax.get_ylim()
        ybound = color_ax.get_ybound()

        ticks = []
        tick_labels = []
        for value in tick_values:
            ticks.append((value - v_min)/(v_max - v_min)*100)
            tick_labels.append(str(value))

        color_ax.set_yticks(ticks)
        color_ax.set_yticklabels(tick_labels)
        color_ax.set_ylim(ylim)
        color_ax.set_ybound(ybound)

        # color_ax.set_ylabel(ylabel)

        right_color_ax = color_ax.twinx()
        right_color_ax.tick_params(labelsize=8)
        ticks = []
        tick_labels = []
        for std_dev in range(-10, 10):
            value = std_dev*baseline_stds[field] + baseline_means[field]
            ticks.append((value - v_min)/(v_max - v_min)*100)

            if std_dev == 0:
                tick_labels.append("0")
            else:
                tick_labels.append("{}$\\sigma$".format(std_dev))

        right_color_ax.set_yticks(ticks)
        right_color_ax.set_yticklabels(tick_labels)
        right_color_ax.set_ylim(color_ax.get_ylim())
        right_color_ax.set_ybound(color_ax.get_ybound())


    with plot_to_file('pen_vs_light_monod/hsa'):
        plot_phase('hsa', tick_values=[1.2, 1.5, 1.8, 2.1, 2.4, 2.7], ylabel='$H$', fig_label='B')
    with plot_to_file('pen_vs_light_monod/cell_height'):
        plot_phase('cell_height', tick_values=[8.1, 8.3, 8.5, 8.7, 8.9, 9.1],
                   ylabel='Mean Biomass Height', fig_label='A')


