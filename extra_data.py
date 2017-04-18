#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
These functions are used to make a usable dataset out of the Udacity data.
'''

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle

image_path = '/home/lucfrachon/Pictures/object-detection-crowdai/'



def make_noncar_boxes(img, patches, y_min = 583, y_max = 980, n_per_img = 6, mean_size = 96):
    '''
    Creates n_per_image random box coordinates per image, that don't contain a car 
    (or more precisely, do not intersect with areas marked as containing a car).
    '''
    boxes = []

    # Make a blank image
    canvas = np.zeros_like(img[:, :, 0])
    car_boxes = patches.loc[:, 'xmin':'ymax']
    img_copy = np.copy(img)
    
    # Make a map of the bounding boxes
    for row in car_boxes.itertuples():
        xmin = min(row[1], row[2])
        xmax = max(row[1], row[2])
        ymin = min(row[3], row[4])
        ymax = max(row[3], row[4])
        # Vertices
        v0 = np.array([xmin, ymin])
        v1 = np.array([xmax, ymin])
        v2 = np.array([xmax, ymax])
        v3 = np.array([xmin, ymax])
        # Draw box on canvas
        cv2.fillConvexPoly(canvas, np.array([v0, v1, v2, v3]), (255.))
        tried, retained = 0, 0

        while len(boxes) < n_per_img:  # Try new random boxes until we have as many as specified
            tried += 1
            # Randomly build boxes
            # Top left corner:
            x0, y0 = np.random.randint(0, 1760), np.random.randint(y_min, y_max)
            # Bottom right corner (size is taken from a truncated normal distribution)
            lower, upper = 32, 160
            sigma = mean_size / 3
            box_size = int(stats.truncnorm((lower - mean_size) / sigma, 
                (upper - mean_size) / sigma, loc = mean_size, 
                scale = sigma).rvs())
            x1, y1 = x0 + box_size, y0 + box_size

            # Extract this patch of the canvas
            box_array = canvas[y0:y1, x0:x1]

            # Make sure this box doesn't intersect with car boxes already present on the canvas:
            if (box_array == 0.).all() and (x1 <= 1920) and (y1 <= y_max):
                # In that case append to list
                boxes.append(((x0, y0), (x1, y1)))
                retained += 1

    return boxes  # Return list of non-car boxes for this image


def extract_patches(dataframe, min_x, n_lines = None, size = (64, 64)):
    '''
    Uses the CSV file provided with the Udacity data to extract picture patches of cars and resize
    them to the specified size.
    The dataframe passed as argument contains box coordinates, frame filenames and labels from
    the images in the dataset.
    '''

    car_imgs = []
    noncar_imgs = []

    # Filter out all non-car patches
    cars_only = dataframe.loc[dataframe['Label'] == 'Car', 
        ['xmin', 'xmax', 'ymin', 'ymax', 'Frame']]  
    # Warning: These column names are wrong! We need to swap two:
    cars_only[['xmax', 'ymin']] = cars_only[['ymin', 'xmax']]

    # Sort the dataframe by 'Frame'
    cars_only.sort_values('Frame', inplace = True)

    if n_lines:
        cars_only = cars_only.iloc[:n_lines, :]

    for filename in cars_only['Frame'].unique():
        img_patches = cars_only[cars_only['Frame'] == filename]
        img = plt.imread(image_path + filename)
        print("Processing file:", filename)
        count = 0
        for row in img_patches.itertuples():
            # Warning: the column names are all scambled up in the CSV file
            xmin = min(row[1], row[2])
            xmax = max(row[1], row[2])
            ymin = min(row[3], row[4])
            ymax = max(row[3], row[4])

            if (xmin < xmax) and (ymin < ymax) and (xmin >= min_x):
                car_img_out = cv2.resize(img[ymin:ymax, xmin:xmax], size)
                car_imgs.append(car_img_out)
                count += 1

        if count != 0:
            # Make a list of random non-car boxes
            noncar_boxes = make_noncar_boxes(img, img_patches, y_min = 450, n_per_img = count)
            # Extract these image patches and append them to noncar_imgs
            for box in noncar_boxes:
                
                x0, y0 = box[0][0], box[0][1]
                x1, y1 = box[1][0], box[1][1]
                noncar_img_out = img[y0:y1, x0:x1]
                
                noncar_imgs.append(cv2.resize(noncar_img_out, (64, 64)))

    return car_imgs, noncar_imgs


if __name__ == '__main__':

    data_frame = pd.read_csv(image_path + 'labels.csv')
    cars, notcars = extract_patches(data_frame, 750)
    with open('udacity_data.p', 'wb') as f:
        pickle.dump(cars, f)
        pickle.dump(notcars, f)

    print("Cars:", len(cars), "Not cars:", len(notcars))

    plt.subplot(121)
    plt.imshow(cars[0])
    plt.subplot(222)
    plt.imshow(cars[23455])
    plt.savefig('cars.png')

    plt.figure()
    plt.subplot(121)
    plt.imshow(cars[0])
    plt.subplot(122)
    plt.imshow(cars[31048])
    plt.savefig('notcars.png')

