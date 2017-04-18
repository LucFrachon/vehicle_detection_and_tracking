#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Defines a Detected_Cars() class that tracks positive detections.
A custom Queue() class is used to store detections.
'''

import numpy as np


class Queue():
    def __init__(self, elt_shape):
        '''
        Initiates an instance of Queue() with element shape = elt_shape
        '''
        self.items = []
        self.elt_shape = elt_shape

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        if self.elt_shape[0] == item.shape[0] and self.elt_shape[1] == item.shape[1]:
            self.items.insert(0, item)
        else:
            raise ValueError("Wrong element shape: Expected", self.elt_shape, ", got", item.shape)

    def dequeue(self):
        self.items.pop()

    def size(self):
        return len(self.items)

    def mean(self):  
        np_queue = np.array(self.items)
        return np.mean(np_queue, 0)

    def median(self):
        np_queue = np.array(self.items)
        return np.median(np_queue, 0)



class Heatmaps(Queue):
    def __init__(self, n_frames, weights, elt_shape):
        Queue.__init__(self, elt_shape)
        self.n_frames = n_frames  # Number of frames to average detections on
        # self.overlay = overlay  # Min. proportion of pixels that two boxes need to have in common in 
        #                         # order to be considered as a single vehicle detection
        if len(weights) == n_frames:
            self.weights = weights  # Weight of each frame in average. 
                                    #Length must be equal to self.n_frames.
        else:
            raise ValueError("The weight vector has incorrect length.")

    def enqueue_dequeue(self, item):
        self.enqueue(item)
        if len(self.items) > self.n_frames:
            self.dequeue()


    def weighted_average(self):
        '''
        Computes the weighted average of all heatmaps containted in the self.heatmap queue with 
        weights provided by self.weights.
        '''
        n_elts = len(self.items)
        if n_elts < self.n_frames:  # If the queue is incomplete (less than n_frames elements):
            # Make copies of the oldest element up to the number of missing elements
            copies = np.tile(self.items[-1], (self.n_frames - n_elts , 1, 1)) 
            heatmap_array = np.stack((*self.items, *copies), axis = 0).astype('float32')
        else:  # Otherwise, just stack the queue elements together
            heatmap_array = np.stack(self.items, axis = 0).astype('float32')

        # Calculate the weighted average of the n_frames heatmaps
        avg_hm = np.average(heatmap_array, axis = 0, weights = self.weights)

        return avg_hm
 




