
import numpy as np
import cv2
import random as rdm

def make_image(imageShape):
    image = np.zeros(imageShape, np.uint8)

    height = np.clip(rdm.gauss(7, 6), 1, 13)
    image[:int(round(height)), :] = 1

    chunks = rdm.randint(3, 8) 
    for _ in range(chunks): delete_chunk(image)

    mask = np.random.random(imageShape) < 0.9
    image[mask] = 0

    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.dilate(image, elem, dst=image)

    return image

def delete_chunk(image):
    center = rdm.randint(0, image.shape[1]-1)
    radius = rdm.randint(8, 24)
    image[:, max(0, center-radius):min(image.shape[1]-1, center+radius)] = 0

def generate(depth, numSamples=100, numCols=256):
    samples = np.zeros((numSamples, depth))
    for i in range(numSamples):
        image = make_image((depth, numCols))
        samples[i, :] = image.sum(axis=1)/float(numCols)
    return samples

def show_example():
    from matplotlib import pyplot as plt
    plt.imshow(make_image())
    plt.show()

def show_curves():
    from matplotlib import pyplot as plt
    plt.plot(generate(20, numSamples=50).T)
    plt.show()
