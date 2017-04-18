#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from feature_extraction import *
#from detection import *
import pickle
import numpy as numpy
import numpy.random as random
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler


def find_cars(img, xstart, xstop, ystart, ystop, color_cspace, hog_cspace, classifier, 
    scaler, prob_thresh = .5, scale = 1., stride = (2, 2), spatial_size = (16, 16), hist_bins = 64, 
    orient = 9, 
    pix_per_cell = 16, cells_per_block = 2, blk_norm = 'L2-Hys', gamma_corr = True,
    spatial_feat = True, hist_feat = True, hog_feat = True):
    '''
    Scans an image to identify cars: first computes the HOGs of the region of interest of the 
    original image, then slides a window accross it to extract HOG, color histogram and spatial
    binnnig features for every patch. Scales the features using the trained scaler, then runs a 
    prediction on each patch to detect cars (using the previously trained classifier).
    Whenever a car is found, the corresponding position of the window is added to the list of 
    'hot' windows.

    '''
    draw_img = np.copy(img)
    check_scale(draw_img)

    all_windows = []
    hot_windows = []
    
    # Crop image to region of interest and apply Gaussian blur
    # img_tosearch = np.clip(np.float32(cv2.blur(img[ystart : ystop, xstart : xstop, :], (5,5))), 
    #     0., 1.)
    img_tosearch = np.float32(img[ystart : ystop, xstart : xstop, :])
    check_scale(img_tosearch)

    # Convert image for color-related features
    to_search_color = convert_color(img_tosearch, cspace = color_cspace, equalize = True)
    # Convert image for HOG features
    to_search_hog = convert_color(img_tosearch, cspace = hog_cspace, equalize = True)

    if scale != 1:
        imshape = to_search_color.shape
        to_search_color = cv2.resize(to_search_color, 
            (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
        to_search_hog = cv2.resize(to_search_hog, 
            (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    hog_ch1 = to_search_hog[:, :, 0]
    hog_ch2 = to_search_hog[:, :, 1]
    hog_ch3 = to_search_hog[:, :, 2]

    # Define blocks and steps as above
    nxcells = hog_ch1.shape[1] // pix_per_cell - 1
    nycells = hog_ch1.shape[0] // pix_per_cell - 1

    # nfeat_per_block = orient * cells_per_block ** 2
    # 64 was the size of the training images, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = window // pix_per_cell - cells_per_block + 1

    # eg. pix_per_cell == 8, window == 64, stride == 2 <==> overlap == 0.75
    nxsteps = (nxcells - nblocks_per_window) // stride[0]
    nysteps = (nycells - nblocks_per_window) // stride[1]
    print("Number of steps x:", nxsteps)
    print("Number of steps y:", nysteps)

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(hog_ch1, orient, pix_per_cell, cells_per_block, feature_vec = False, 
        blk_norm = blk_norm, gamma_corr = gamma_corr)
    hog2 = get_hog_features(hog_ch2, orient, pix_per_cell, cells_per_block, feature_vec = False, 
        blk_norm = blk_norm, gamma_corr = gamma_corr)
    hog3 = get_hog_features(hog_ch3, orient, pix_per_cell, cells_per_block, feature_vec = False,
        blk_norm = blk_norm, gamma_corr = gamma_corr)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * stride[1]
            xpos = xb * stride[0]
            if hog_feat:  # If HOG features are required by the trained model:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos : ypos + nblocks_per_window, 
                                 xpos : xpos + nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos : ypos + nblocks_per_window, 
                                 xpos : xpos + nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos : ypos + nblocks_per_window, 
                                 xpos : xpos + nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            if spatial_feat or hist_feat:  # If either color feature is required by the model:
                # Extract the image patch
                subimg = cv2.resize(to_search_color[ytop : ytop + window, 
                                                xleft : xleft + window], (window, window))
          
                if spatial_feat:
                    # Get spatial features
                    spatial_features = bin_spatial(subimg, size = spatial_size)
                else:
                    spatial_features = None
                if hist_feat:
                    # Get histogram features
                    hist_features = color_hist(subimg, nbins = hist_bins, bins_range = (0., 1.))
                else:
                    hist_features = None

                # Scale features and make a prediction
                reshaped_feat = np.hstack((spatial_features, hist_features, 
                    hog_features)).reshape(1, -1)
                test_features = scaler.transform(reshaped_feat)
            else:
                test_features = scaler.transform(hog_features)
            
            test_prob = classifier.predict_proba(test_features)
            prediction = classifier.predict(test_features)

            xbox_left = np.int(xstart + xleft * scale)
            ytop_draw = np.int(ystart + ytop * scale)
            win_draw = np.int(window * scale)
            all_windows.append(((xbox_left, ytop_draw),
                             (xbox_left + win_draw, ytop_draw + win_draw)))

            if test_prob[0][1] >= prob_thresh:
                print("Prob:", test_prob[0][1])
                hot_windows.append(((xbox_left, ytop_draw),
                             (xbox_left + win_draw, ytop_draw + win_draw)))

    return all_windows, hot_windows


def scan_at_scales(img, xstart_stop, ystart_stop, clf, scaler, prob_thresh = 0.5, scales = (1.), 
    color_cspace = 'HLS', 
    hog_cspace = 'YCrCb', **kwargs):
    ''' 
    Takes an image 'img' and a list of 'scales', and runs find_cars() on the image at all the scales
    in 'scales', with the specified arguments.
    Returns the list of boxes where a vehicle was detected.
    '''

    all_windows = []
    hot_windows = []

    for i, scale in enumerate(scales):
        print('Scanning at scale', scale)
        xstart = xstart_stop[i][0]
        xstop = xstart_stop[i][1]
        ystart = ystart_stop[i][0]
        ystop = ystart_stop[i][1]

        a_wdws, h_wdws = find_cars(img, xstart, xstop, ystart, ystop, 'HLS', 'YCrCb', clf, 
            scaler, prob_thresh = prob_thresh, scale = scale, **kwargs)
        all_windows.extend(a_wdws)
        hot_windows.extend(h_wdws)

    return all_windows, hot_windows


if __name__ == '__main__':
    img = read_and_scale_image('./test_images/test6.jpg')

    with open('scaler.p', 'rb') as f:
        scaler = pickle.load(f)
    with open('classifier.p', 'rb') as g:
        clf = pickle.load(g)

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
    windows, _ = scan_at_scales(img, xstart_stop, ystart_stop, clf, scaler, 0.99, 
        scales = (0.70, 1., 2.), color_cspace = 'HLS', hog_cspace = 'YCrCb', 
        stride = (2, 2), spatial_size = (16, 16), hist_bins = 64, orient = 9, 
        pix_per_cell = 16, cells_per_block = 2, blk_norm = 'L2-Hys', gamma_corr = True,
        spatial_feat = False, hist_feat = False, hog_feat = True)

    for wdw in windows:
        cv2.rectangle(img, wdw[0], wdw[1], (random.randint(0, 255)/255,random.randint(0, 255)/255,random.randint(0, 255)/255))

    f = plt.figure(figsize = (22, 15))
    plt.imshow(img)
    f.tight_layout()
    plt.savefig('sliding_window.png')

