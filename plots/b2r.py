
from biofilm import util
util.set_h5(util.results_h5_path('b2r'))

from biofilm.model import analysis as an
from biofilm.model.spec import make_query

from plots.util import plot_to_file

from matplotlib import pyplot as plt
import numpy as np
import cv2


def compute():
    an.mass.compute_specs(recompute=True)
    #an.convex_density.compute_specs(recompute=True)
    #an.mean_cell_height.compute_specs(recompute=True)
    #an.heights.compute_specs(recompute=True)
    an.perimeter.compute_specs(recompute=True)
    #an.roughness.compute_specs(recompute=True)
    #an.horizontal_surface_area.compute_specs(recompute=True)
    an.b2r.compute_specs()
    #an.br.compute_specs()
    #an.growth_time.compute_specs(recompute=True)
    #an.recip_b2r.compute_specs()
    pass
#compute()

query = '(boundary_layer>=1)&(boundary_layer<=8)'

if False:
    b2r, perimeter = an.b2r.scatter_plot(an.perimeter, spec_query=query, 
                                         statistic='all')
    np.savez('plots/images/b2r/b2r.npz', b2r=b2r, perimeter=perimeter)
    plt.clf()


# B2R MEAN CELL HEIGHT
if True:
    data = np.load('plots/images/b2r/b2r.npz')

    from matplotlib.gridspec import GridSpec

    with plot_to_file('b2r/b2r_vs_perimeter'):
        plt.figure(figsize=(3.27, 1.15*3.27))

        rows = 9
        gs1 = GridSpec(rows, 1)
        gs1.update(top=1.0, bottom=0.15)
        gs2 = GridSpec(rows, 1)
        gs2.update(top=1.0, bottom=0.0, hspace=0.05)

        plt.subplot(gs1[:rows-2, 0])
        plt.plot(data['b2r'], data['perimeter'], 'b.', ms=3.0, mec='none', 
                 mfc=(0, 0, 1.0, 0.33))
        plt.xlim([0, 30])
        plt.ylim([2, 9])
        plt.xlabel('$1/n_s$')
        plt.ylabel('Perimeter')

        plt.hold(True)

        # Image labels
        a_loc = 0.188*4**2, 3.128
        # plt.plot(a_loc[0], a_loc[1], 'ko', ms=8.0, mew=1.5, mfc='none')
        plt.annotate('A', a_loc, (6, 2.3), size=12, weight='bold',
                     arrowprops={'facecolor':'black', #'shrink': 0.1,
                         'arrowstyle':'->', 'linewidth':1.5})

        b_loc = 0.561*7**2, 6.531
        # plt.plot(a_loc[0], a_loc[1], 'ko', ms=8.0, mew=1.5, mfc='none')
        plt.annotate('B', b_loc, (25, 4.25), size=12, weight='bold',
                     arrowprops={'facecolor':'black', #'shrink': 0.1,
                         'arrowstyle':'->', 'linewidth':1.5})

        def show_image(label, name):
            image = cv2.imread('model_images/b2r/' + name)
            image[:, :, 0] = image[:, :, 2]
            image[:, :, 2] = 0
            plt.imshow(image)

            plt.text(-25, 64, label, weight='bold', size=12, va='center', ha='right')

            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)

        plt.subplot(gs2[rows-2, 0])
        show_image('A', 'low/boundary_layer4.000-media_ratio0.188-perimeter3.126-0.png')

        plt.subplot(gs2[rows-1, 0])
        show_image('B', 'high/boundary_layer7.000-media_ratio0.561-perimeter6.531-2.png')

        # plt.tight_layout()





