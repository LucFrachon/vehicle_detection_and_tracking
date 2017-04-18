#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import *
from detection import *
import pickle
import numpy as numpy
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


if __name__ == '__main__':
    with open('udacity_data.p', 'rb') as f:
        cars = pickle.load(f)
        notcars = pickle.load(f)

    indices = random.randint(0, len(cars), 4)

    f, axes = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 20))

    for i in range(4):
        axes[i, 0].imshow(cars[indices[i]])
        axes[i, 0].set_title("Car")
        axes[i, 1].imshow(notcars[indices[i]])
        axes[i, 1].set_title("Not Car")

    f.tight_layout()
    plt.savefig('udacity_images.png')