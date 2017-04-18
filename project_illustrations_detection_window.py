#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import *
from detection import *
import pickle
import numpy as numpy
from classes import Heatmaps
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler



if __name__ == '__main__':
    with open('scaler.p', 'rb') as f:
        scaler = pickle.load(f)
    with open('classifier.p', 'rb') as g:
        clf = pickle.load(g)
    imgs = []


    for i in range(6):
        imgs.append(read_and_scale_image('./test_images/test' + str(i + 1) + '.jpg'))

    f, axes = plt.subplots(nrows = 6, ncols = 2, figsize = (9, 35))

    for i, img in enumerate(imgs):
        img_size = (img.shape[1], img.shape[0])
        xstart_stop = [
                   (int(img_size[0] * 0.), int(img_size[0] * 1.)), 
                   (int(img_size[0] * 0.), int(img_size[0] * 1.)), 
                   (int(img_size[0] * 0.), int(img_size[0] * 1.))
                   ]
        ystart_stop = [
                   (int(img_size[1] * 0.54), int(img_size[1] * 0.75)),
                   (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   (int(img_size[1] * 0.54), int(img_size[1] * 1.))
                   ]
        heatmap = scan(img, xstart_stop, ystart_stop, clf, scaler, 0.999, 
            scales = (0.70, 1., 2.), color_cspace = 'HLS', hog_cspace = 'YCrCb', 
            stride = (2, 2), spatial_size = (16, 16), hist_bins = 64, orient = 9, 
            pix_per_cell = 16, cells_per_block = 2, blk_norm = 'L2-Hys', gamma_corr = True,
            spatial_feat = False, hist_feat = False, hog_feat = True)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(heatmap)

    f.tight_layout()
    plt.savefig('heatmaps.png')

