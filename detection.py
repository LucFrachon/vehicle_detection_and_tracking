#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
These functions take a trained classifier and an image (can be used on clips too with
moviepy.editor.VideoFileClip.fl_image()) and output a copy of the image with boxes around detected
cars.
'''

from feature_extraction import *
import pickle
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from classes import Heatmaps


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

    hot_windows = []
    
    # Crop image to region of interest and apply Gaussian blur
    # img_tosearch = np.clip(np.float32(cv2.blur(img[ystart : ystop, xstart : xstop, :], (5,5))), 
    #     0., 1.)
    img_tosearch = np.float32(img[ystart : ystop, xstart : xstop, :])
    check_scale(img_tosearch)

    # Convert image for color-related features
    to_search_color = convert_color(img_tosearch, cspace = color_cspace, equalize = False)
    # Convert image for HOG features
    to_search_hog = convert_color(img_tosearch, cspace = hog_cspace, equalize = False)

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
            
            if test_prob[0][1] >= prob_thresh:
                print("Prob:", test_prob[0][1])
                xbox_left = np.int(xstart + xleft * scale)
                ytop_draw = np.int(ystart + ytop * scale)
                
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left, ytop_draw),
                             (xbox_left + win_draw, ytop_draw + win_draw)))     

    return hot_windows


def scan_at_scales(img, xstart_stop, ystart_stop, clf, scaler, prob_thresh = 0.5, scales = (1.), 
    color_cspace = 'HLS', 
    hog_cspace = 'YCrCb', **kwargs):
    ''' 
    Takes an image 'img' and a list of 'scales', and runs find_cars() on the image at all the scales
    in 'scales', with the specified arguments.
    Returns the list of boxes where a vehicle was detected.
    '''

    hot_windows = []

    for i, scale in enumerate(scales):
        print('Scanning at scale', scale)
        xstart = xstart_stop[i][0]
        xstop = xstart_stop[i][1]
        ystart = ystart_stop[i][0]
        ystop = ystart_stop[i][1]

        wdws = find_cars(img, xstart, xstop, ystart, ystop, 'HLS', 'YCrCb', clf, 
            scaler, prob_thresh = prob_thresh, scale = scale, **kwargs)
        hot_windows.extend(wdws)

    return hot_windows


def add_heat(heatmap, bbox_list):
    '''
    Takes an existing heatmap (1-channel image) and a list of boxes, then increments values of all
    pixels within any of these boxes.
    Returns the updated heatmap.
    '''
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1.
        
    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    '''
    Turns all pixels within heatmap that are below threshold to zero.
    '''
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



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


def draw_labeled_bboxes(img, heatmap):
    '''
    Takes a set of labels (output from scipy.ndimage.measurements.label()) and an image and draws
    colored boxes around the areas corresponding to each detected car.
    Retuns the image with boxes drawn around detections and the x and y coordinates of all pixels 
    for each label.
    '''

    labels = label(heatmap)
    max_label = labels[1] + 1
    img_out = np.uint8(np.copy(img) * 255)

    label_coords = []
    # Iterate through all detected cars
    for car_number in range(1, max_label):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img_out, bbox[0], bbox[1], (0., 0., 255), 3)
        # Add this box to the list
        label_coords.append((car_number, nonzerox, nonzeroy))
    # Return the image
    return img_out, label_coords, heatmap


def movie_pipeline(img, xstart_stop, ystart_ystop, clf, scaler, prob_thresh, scales, hm_thresh, 
    **kwargs):
    '''
    Wrapper function implementing the pipeline and passed to VideoFileClip.fl_image() to process
    video clips.
    '''
    img = np.float32(img) / 255.
    _ = make_heatmap_from_frame(img, xstart_stop, ystart_stop, clf, scaler, prob_thresh, scales, 
        **kwargs)
    thresholded_avg_hm = apply_threshold(heatmaps.weighted_average(), hm_thresh)
    img_out, _, heatmap = draw_labeled_bboxes(img, thresholded_avg_hm) 
    heatmap = heatmap.clip(0., 255.).astype('uint8')
    inserted_img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2GRAY).astype('uint8')
    inserted_img = cv2.resize(cv2.addWeighted(inserted_img, 1., heatmap, 80., 0. ), (444, 250))

    img_out[:250, :444] = np.stack((inserted_img, inserted_img, inserted_img), axis = 2)
    return img_out
    

if __name__ == '__main__':
    # Load trained scaler and classifier
    with open('scaler.p', 'rb') as f:
        scaler = pickle.load(f)
    with open('classifier.p', 'rb') as g:
        clf = pickle.load(g)

    # Load a test image
    img = read_and_scale_image('./test_images/test6.jpg')
    img_size = (img.shape[1], img.shape[0])
    # Scales to scan the image at:
    scales = (0.70,
              # 1.1,
              1.,
              # 1.8,
              2.0,
              # 2.9,
              # 3.4,
              # 4.0
              )

    # Region of interest for each scale:
    xstart_stop = [
                   (int(img_size[0] * 0.), int(img_size[0] * 1.)), 
                   (int(img_size[0] * 0.), int(img_size[0] * 1.)), 
                   (int(img_size[0] * 0.), int(img_size[0] * 1.)), 
                   # (int(img_size[0] * 0.), int(img_size[0] * 1.)),
                   # (int(img_size[0] * 0.), int(img_size[0] * 1.)),
                   # (int(img_size[0] * 0.), int(img_size[0] * 1.)),
                   # (int(img_size[0] * 0.), int(img_size[0] * 1.)),
                   # (int(img_size[0] * 0.), int(img_size[0] * 1.))
                   ]
    ystart_stop = [
                   (int(img_size[1] * 0.54), int(img_size[1] * 0.75)),
                   (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   (int(img_size[1] * 0.54), int(img_size[1] * 1.)), 
                   # (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   # (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   # (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   # (int(img_size[1] * 0.54), int(img_size[1] * 1.)),
                   # (int(img_size[1] * 0.54), int(img_size[1] * 1.))
                   ]

    # Parameters affecting feature vector size:
    spatial_size = (16, 16)
    hist_bins = 64
    orient = 9
    pix_per_cell = 16
    cells_per_block = 2

    # Heatmap threshold:
    stride = (2, 2)
    hm_thresh = .6

    # Positive classification threshold:
    prob_thresh = .9995  # Probability of belonging to positive class

    # Create a heatmap queue:
    q_length = 15
    heatmaps = Heatmaps(q_length, [1. for i in range(q_length)], (720, 1280))

    # Process video clip
    clip_in = VideoFileClip('project_video.mp4')
    clip_out = clip_in.fl_image(lambda x: movie_pipeline(x, xstart_stop, ystart_stop, clf, scaler, 
        prob_thresh, scales, hm_thresh, stride = stride, color_cspace = 'HLS', hog_cspace = 'YCrCb', 
        spatial_size = spatial_size, hist_bins = hist_bins, orient = orient, 
        pix_per_cell = pix_per_cell, cells_per_block = cells_per_block, blk_norm = 'L2-Hys', 
        gamma_corr = True, spatial_feat = False, hist_feat = False, hog_feat = True))
    clip_out.write_videofile('project_video_result.mp4', audio = False)