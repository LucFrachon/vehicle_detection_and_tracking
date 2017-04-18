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

def make_heatmap_from_frame(img, xstart_stop, ystart_stop, clf, scaler, prob_thresh, scales, 
    **kwargs):  # kwargs to be passed on to find_cars()
    '''
    Takes a single image and calls 'scan_at_scales()' to find all hot windows. Then uses 
    'add_heat()' and 'apply_threshold()' to make a thresholded heatmap. Add this heatmap to the 
    existing heatmap queue.
    Returns the thresholded heatmap.
    '''
    # Look for cars in the test image
    hot_windows = scan_at_scales(img, xstart_stop, ystart_stop, clf, scaler, 
        prob_thresh = prob_thresh,
        scales = scales, 
        **kwargs)

    heat = np.zeros_like(img[:, :, 0]).astype('float32')
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Visualize the heatmap when displaying    
    # heat = np.clip(heat, 0., 1.)
    # check_scale(heat, (0., 1.), 'float64')

    # Add this heatmap to the queue
    heatmaps.enqueue_dequeue(heat)

    return heat  # Note: this heatmap's values are positive but have no upper bound.


if __name__ == '__main__':
    with open('scaler.p', 'rb') as f:
        scaler = pickle.load(f)
    with open('classifier.p', 'rb') as g:
        clf = pickle.load(g)
    imgs = []
    q_length = 3

    img_indices = [2, 3, 6, 4, 1, 5]
    heatmaps = Heatmaps(q_length, [1. for i in range(q_length)], (720, 1280))

    for i in range(6):
        imgs.append(read_and_scale_image('./test_images/test' + str(i + 1) + '.jpg'))

    f, axes = plt.subplots(nrows = 6, ncols = 3, figsize = (12, 15))

    for i, idx in enumerate(img_indices):
        print(idx)
        img = imgs[idx - 1]
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
        heatmap = make_heatmap_from_frame(img, xstart_stop, ystart_stop, clf, scaler, 0.999, 
            scales = (0.70, 1., 2.), color_cspace = 'HLS', hog_cspace = 'YCrCb', 
            stride = (2, 2), spatial_size = (16, 16), hist_bins = 64, orient = 9, 
            pix_per_cell = 16, cells_per_block = 2, blk_norm = 'L2-Hys', gamma_corr = True,
            spatial_feat = False, hist_feat = False, hog_feat = True)
        thresholded_hm = apply_threshold(heatmaps.weighted_average(), 0.4)
        img_out, _ , _ = draw_labeled_bboxes(img, thresholded_hm)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original image')
        axes[i, 1].imshow(thresholded_hm, cmap = 'hot')
        axes[i, 1].set_title('Thresholded avg heatmap')
        axes[i, 2].imshow(img_out)
        axes[i, 2].set_title('Bounding boxes')

    f.tight_layout()
    plt.savefig('bounding_boxes.png')

