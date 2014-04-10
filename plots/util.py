

from contextlib import contextmanager
from matplotlib import pyplot as plt
import os

IMAGES = 'plots/images'

@contextmanager
def plot_to_file(name):
    plt.clf()
    yield
    plt.savefig(os.path.join(IMAGES, name + '.png'))
