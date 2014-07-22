

from contextlib import contextmanager
from matplotlib import pyplot as plt
import os

from matplotlib import rc
rc('font', family='Arial', size='10.0')
rc('figure', dpi=400, figsize=(6.83, 6.83/2))
rc('savefig', bbox='tight', pad_inches=0.03) # roughly 2pts
rc('text', usetex=False)

IMAGES = 'plots/images'

@contextmanager
def plot_to_file(name):
    plt.clf()
    yield
    plt.savefig(os.path.join(IMAGES, name + '.tiff'), dpi=400)
