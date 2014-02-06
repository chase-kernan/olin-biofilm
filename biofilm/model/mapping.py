
import numpy as np

__all__ = ['make_sin_sigmoid', 'make_monod']

def make_richard_sigmoid(max_growth, growth_rate, balance, q=0.1):
    """Richard's curve (generalized logistic function).

    http://en.wikipedia.org/wiki/Generalised_logistic_function"""
    #k = 1 + q
    #q = -1 + (k/y0)^v
    #(q + 1)^(1/v) = k/y0
    #y0 = (1 + q)/(1 + q)^(1/v)
    #y0 = (1 + q)^(1 - 1/v)

    recip_b = 1./balance
    offset = (1 + q)**(1 - recip_b)
    k = 1 + q
    def sigmoid(x):
        return k/(1 + q*np.exp(-growth_rate*(x-max_growth)))**recip_b - offset
    return sigmoid

def make_sin_sigmoid(center, radius):
    freq = np.pi/(2*radius)
    def sigmoid(x):
        return 0.5 + 0.5*np.sin(freq*np.clip(x-center, -radius, radius))
    return sigmoid

def make_monod(k):
    def monod(x):
        return x/(x + k)
    return monod

def monod_of_array(xs, k):
    """NOTE: this modifies xs!"""
    xs /= xs + k
    return xs

def test():
    from matplotlib import pyplot as plt
    x = np.linspace(0, 1, 1000)

    plt.subplot(2, 1, 1)
    plt.hold(True)
    for radius in [0.01, 0.1, 0.2, 0.4]:
        for center in np.linspace(0.3, 0.7, 5):
            plt.plot(x, make_sin_sigmoid(center, radius)(x))

    plt.subplot(2, 1, 2)
    plt.hold(True)
    for k in [0.01, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0]:
        plt.plot(x, make_monod(k)(x))

    plt.show()

if __name__ == '__main__':
    test()

