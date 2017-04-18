#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
These functions are used to read an image and extract its features (or a portion of an
image) for training and prediction purposes.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage.feature import hog
import glob
import pickle


def check_scale(array, target = (0., 1.), d_type = 'float32'):
    '''
    A debugging function that can be used to check if an image 'array' is scaled
    from 0. to 1. (or any other boundaries, defined in 'target') with the data
    type specified in 'd_type'.
    It throws a ValueError and a message if one of these conditions is not met, 
    and returns the minimum and maximum values and data type of the array if the
    conditions are met.
    '''
    t = np.zeros(1, dtype = d_type)
    if (np.min(array) < target[0] or np.max(array) > target[1]) or \
        t.dtype != array.dtype:
        print("Array range and type:", (np.min(array), np.max(array)),
              array.dtype)
        raise ValueError("The scale or dtype of the array is wrong.")
    else:
        return np.min(array), np.max(array), array.dtype
        

def read_and_scale_image(filename, d_type = 'float32'):
    '''
    Read an image and scale it to [0., 1.] (or any other interval) if it is not 
    already the case. It also converts it to the specified data type.
    - filename: A path name to the image file
    - target: Target scale
    - d_type: Target data type
    '''
    img = mpimg.imread(filename)
    if np.max(img) > 1.:
        img = img.astype(d_type) / 255.
    return img


def convert_color(img, cspace = 'same', equalize = False):
    '''
    Takes an RGB image array 'img' and converts it to the color space specified in cspace.
    Returns the converted image or a copy of the original image if no conversion is specified (ie.
    'cspace == 'RGB' or 'same'). All channels scaled to [0., 1.].
    '''
    flags = {'HSV': cv2.COLOR_RGB2HSV, 'HLS': cv2.COLOR_RGB2HLS,
             'LUV': cv2.COLOR_RGB2LUV, 'YUV': cv2.COLOR_RGB2YUV, 
             'YCrCb': cv2.COLOR_RGB2YCrCb}

    if equalize:
        img_yuv = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2YUV) * 255.)
        img_yuv[:,:, 0] = cv2.equalizeHist(img_yuv[:,:, 0])
        img = np.float32(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) / 255.)

    check_scale(img)
    if cspace == 'RGB' or cspace == 'same':
        img_conv = np.copy(img).astype('float32')
    elif cspace == 'HSV' or cspace == 'HLS':
        img_conv = cv2.cvtColor(img, flags[cspace]).astype('float32')
        img_conv[:, :, 0] /= 360.
    elif cspace == 'LUV' or cspace == 'YUV':
        img_conv = np.maximum(0., cv2.cvtColor(img, flags[cspace]).astype('float32'))
    elif cspace == 'YCrCb':
        img_conv = cv2.cvtColor(img, flags[cspace]).astype('float32')

    check_scale(img_conv, (0., 1.), 'float32')
    return img_conv


def color_hist(img, nbins = 32, bins_range = (0., 1.)):
    '''
    Takes an image 'img' in any 3-channel color space, a number of bins 'nbins' and a bins range 
    'bins_range' (which should correspond to the range of the image array's values, either 
    [0., 1.] or [0, 255]).
    Computes the histograms for each channel with the specified number of bins.
    Returns a feature vector of all the concatenated histogram values.
    '''
    # Compute the histogram of each channel separately
    hist_0 = np.histogram(img[:,:, 0], bins = nbins, range = bins_range)
    hist_1 = np.histogram(img[:,:, 1], bins = nbins, range = bins_range)
    hist_2 = np.histogram(img[:,:, 2], bins = nbins, range = bins_range)

    # Concatenate into a feature vector
    hist_features = np.concatenate((hist_0[0], hist_1[0], hist_2[0]))
    # Return the feature vector
    return hist_features


def bin_spatial(img, size = (32, 32)):
    '''
    Takes an image 'img' and a target 'size' and resizes it to the specified size.
    It then flattens the resulting array into a 1d vector and return this.
    '''
    check_scale(img, (0., 1.))  # Make sure the image is scaled [0., 1.]        
    # Use cv2.resize().ravel() to create the feature vector
    feat_vect = cv2.resize(img, size).ravel()
    return feat_vect


def get_hog_features(img, orient, pix_per_cell, cells_per_block, vis = False, feature_vec = True, 
                     blk_norm = 'L2-Hys', gamma_corr = True):
    '''
    Takes an image 'img' and HOG parameters. 'vis' indicates whether we should return the visualisa-
    tion of the HOG, 'features_vec' whether the output should be flattened into a 1d vector or
    returned as a 5d array (ny_blocks, n_xblocks, cell_per_block, cell_per_block, orient).

    Returns a 1d or 5d array.
    '''

    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        check_scale(img)
        features, hog_image = hog(img, orientations = orient, 
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cells_per_block, cells_per_block),
                                  visualise = vis, feature_vector = feature_vec, 
                                  block_norm = blk_norm, transform_sqrt = gamma_corr)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        check_scale(img)
        features = hog(img, orientations = orient, 
                       pixels_per_cell = (pix_per_cell, pix_per_cell),
                       cells_per_block = (cells_per_block, cells_per_block),
                       visualise = vis, feature_vector = feature_vec, 
                       block_norm = blk_norm, transform_sqrt = gamma_corr)
        return features

def extract_features_from_images(imgs, every_n = 1, color_cspace = 'HLS', hog_cspace = 'YUV', 
    spatial_size = (32, 32), hist_bins = 64, orient = 9, pix_per_cell = 8, cells_per_block = 4,
    hog_channel = 'all', blk_norm = 'L2-Hys',
    gamma_corr = True, spatial_feat = True, hist_feat = True, hog_feat = True):
    '''
    Extracts a list of feature vectors from a list of images. These images are first converted to
    the specified color spaces (can be different for color-related features and HOG features).
    - imgs:         A list of images
    - color_cspace: Color space to convert each image to for spatial binning and color histogram
    - hog_cspace:   Color space to convert each image to for the HOG
    - spatial_size: Spatial size of the spatially binned image
    - hist_bins:    Number of bins (features) of the color-histogram
    - orient:       Number of orientations to bin gradient directions into when building the HOG
    - pix_per_cell: Horizontal and vertical size of each cell used in the HOG transformation
    - cells_per_block: Horizontal and vertical number of cells in each of the blocks used to 
                    normalize the HOG.
    - hog_channel:  Channel(s) on which to compute the HOG. Can be 0, 1, 2 or 'all'
    - blk_norm:     Method used for block normalization. 'L1', 'L2', 'L2-Hys' or None
    - gamma_corr:   Whether to apply gamma (square root) correction to each image before computing
                    the HOG.
    - spatial_feat: Extract spatial features?
    - hist_feat:    Extract color histogram features?
    - hog_feat:     Extract HOG features?

    Returns:        A 1d vector of features extracted from the image.
    '''
    
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images - only select one in 5 images to break time series tracks
    for idx in range(0, len(imgs), every_n):
        file_features = []
        # Read in each one by one
        if isinstance(imgs[idx], str):
            image = read_and_scale_image(imgs[idx])
        else:
            image = (imgs[idx] / 255.).astype('float32')

        # Apply color conversion for color-related features
        c_feature_image = convert_color(image, color_cspace)

        # Apply color conversion for HOG features
        h_feature_image = convert_color(image, hog_cspace)        

        check_scale(c_feature_image)  # cv2 sometimes rescales images to smthg other than (0., 1.)
        check_scale(h_feature_image)

        if spatial_feat == True:
            # Apply bin_spatial()
            spatial_features = bin_spatial(c_feature_image, size = spatial_size)
            file_features.append(spatial_features)
        
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(c_feature_image, nbins = hist_bins, bins_range = (0., 1.))
            file_features.append(hist_features)
            
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'all':
                hog_features = []
                for channel in range(h_feature_image.shape[2]):
                    hog_features.append(get_hog_features(h_feature_image[:,:, channel], 
                                        orient, pix_per_cell, cells_per_block, 
                                        vis = False, feature_vec = True, 
                                        blk_norm = blk_norm, gamma_corr = gamma_corr))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(h_feature_image[:,:, hog_channel], orient, 
                            pix_per_cell, cells_per_block, vis = False, feature_vec = True,
                            blk_norm = blk_norm, gamma_corr = gamma_corr)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


if __name__ == '__main__':
    cars = glob.glob('./vehicles/*/*.png')
    notcars = glob.glob('./non-vehicles/*/*.png')

    cars_small = glob.glob('./vehicles_smallset/*/*.jpeg')
    notcars_small = glob.glob('./non-vehicles_smallset/*/*.jpeg')

    with open("udacity_data.p", 'rb') as f:
        udacity_cars = pickle.load(f)
        udacity_notcars = pickle.load(f)

    print("Data load complete")

    # Parameters affecting feature vector size:
    spatial_size = (16, 16)
    hist_bins = 64 #64
    orient = 9 #9
    pix_per_cell = 16 #8
    cells_per_block = 2 #4

    # Make features arrays
    print("Base dataset:")
    base_car_feat = extract_features_from_images(
        cars, 1, color_cspace = 'HLS', hog_cspace = 'YCrCb', spatial_size = spatial_size,
        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell, 
        cells_per_block = cells_per_block, hog_channel = 'all', blk_norm = 'L2-Hys', 
        gamma_corr = True, spatial_feat = False, hist_feat = False, hog_feat = True) 
    udacity_car_feat = extract_features_from_images(
        udacity_cars, 1, color_cspace = 'HLS', hog_cspace = 'YCrCb', spatial_size = spatial_size,
        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell, 
        cells_per_block = cells_per_block, hog_channel = 'all', blk_norm = 'L2-Hys', 
        gamma_corr = True, spatial_feat = False, hist_feat = False, hog_feat = True) 

    print("Udacity dataset:")
    base_notcar_feat = extract_features_from_images(
        notcars, 1, color_cspace = 'HLS', hog_cspace = 'YCrCb', spatial_size = spatial_size,
        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell, 
        cells_per_block = cells_per_block, hog_channel = 'all', blk_norm = 'L2-Hys', 
        gamma_corr = True, spatial_feat = False, hist_feat = False, hog_feat = True)
    udacity_notcar_feat = extract_features_from_images(
        udacity_notcars, 1, color_cspace = 'HLS', hog_cspace = 'YCrCb', spatial_size = spatial_size,
        hist_bins = hist_bins, orient = orient, pix_per_cell = pix_per_cell, 
        cells_per_block = cells_per_block, hog_channel = 'all', blk_norm = 'L2-Hys', 
        gamma_corr = True, spatial_feat = False, hist_feat = False, hog_feat = True) 

    car_feat = np.vstack((base_car_feat, udacity_car_feat))
    notcar_feat = np.vstack((base_notcar_feat, udacity_notcar_feat))


    with open('features.p', 'wb') as f:
        pickle.dump(car_feat, f)
        pickle.dump(notcar_feat, f)

    print("Feature extraction complete")