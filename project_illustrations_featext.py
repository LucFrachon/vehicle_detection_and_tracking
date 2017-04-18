#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import *
from detection import *
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


if __name__ == '__main__':
    with open('udacity_data.p', 'rb') as f:
        cars = pickle.load(f)
        notcars = pickle.load(f)

    car_idx = random.randint(0, len(cars))
    car_img = (cars[car_idx] / 255.).astype('float32')
    notcar_idx = random.randint(0, len(notcars))
    notcar_img = (notcars[notcar_idx] / 255.).astype('float32')

    plt.imshow(car_img)
    plt.savefig('car_image.png')

    car_hls = convert_color(car_img, 'HLS')
    car_y = convert_color(car_img, 'YCrCb')

    feat_hog0, car_hog0 = get_hog_features(car_y[:,:,0], 9, 16, 2, True)
    feat_hog1, car_hog1 = get_hog_features(car_y[:,:,1], 9, 16, 2, True)
    feat_hog2, car_hog2 = get_hog_features(car_y[:,:,2], 9, 16, 2, True)

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows = 3, ncols = 3, 
        figsize = (20, 20))

    ax1.imshow(car_hls[:,:,0], cmap = 'gray')
    ax1.set_title('HLS channel 0')
    
    ax2.imshow(car_hls[:,:,1], cmap = 'gray')
    ax2.set_title('HLS channel 1')
    
    ax3.imshow(car_hls[:,:,2], cmap = 'gray')
    ax3.set_title('HLS channel 2')

    ax4.imshow(car_y[:,:,0], cmap = 'gray')
    ax4.set_title('YCrCb channel 0')
    
    ax5.imshow(car_y[:,:,1], cmap = 'gray')
    ax5.set_title('YCrCb channel 1')
    
    ax6.imshow(car_y[:,:,2], cmap = 'gray')
    ax6.set_title('YCrCb channel 2')

    ax7.imshow(car_hog0, cmap = 'gray')
    ax7.set_title('HOG channel 0')
    
    ax8.imshow(car_hog1, cmap = 'gray')
    ax8.set_title('HOG channel 1')
    
    ax9.imshow(car_hog2, cmap = 'gray')
    ax9.set_title('HOG channel 2')

    plt.tight_layout()
    plt.savefig('feature_extraction.png')


    spatial = bin_spatial(car_hls, (16, 16))
    hist = color_hist(car_hls, 64, (0., 1.))

    features = np.concatenate((feat_hog0, feat_hog1, feat_hog2, spatial, hist))
    scaler = RobustScaler().fit(features)
    scaled_features = scaler.transform(features)
    
    f = plt.figure()
    plt.plot(scaled_features)
    plt.savefig('feature_plot.png')