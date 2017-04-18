# Vehicle Detection and Tracking

This project implements a pipeline to detect and track vehicles by combining computer vision and machine learning techniques.

---

## Key steps

The pipeline follows the following steps:

1. Extract visual features from a set of images. For each image:
	a. Convert to the HLS color space
	b. **Spatial binning:** Reduce image sizes to 16x16 in each of the 3 channels	
	c. **Histograms of colors:** Bin pixels by their values in each of the 3 color channels, using 64 bins
	d. Convert image to the YCrCb color space
	e. Extract HOG features ([Histogram of Oriented Gradients](http://www.learnopencv.com/histogram-of-oriented-gradients/)) in each of the 3 channels
	f. Unravel and concatenate all these features into a 1d vector
	
2. Train a classifier on these features:
	a. Normalize the vector using scikit-learn's RobustScaler
	b. Divide the data into train, validation and test sets (60 / 40 / 40%) at random
	c. Train a multi-layer perceptron neural network (an SVM classifier is also implemented but is far less efficient and does not scale well with large amounts of data)
	c. Predict on the validation set, extract the wrong predictions and inject them back into the training set (**hard negative mining**)
	d. Re-train using the augmented train set
	e. Dump the trained scaler and classifier to a Pickle file.

3. Sliding window search: For each frame of the project video clip, for each specified scale:
	a. Isolate the region of interest and resize it to $1/scale$
	b. Store versions of the croped image converted into the HLS and YCrCb spaces
	c. Calculate the HOG of the YCrCb version
	d. Run a sliding window over both the HLS image and HOG and extract HOG, spatially binned and color histogram features
	e. Scale the feature vector, scale it using the trained scaler and predict for presence / absence of a car within the window
	f. Save coordinates of 'hot' windows, ie. windows containing a car according to the classifier
	
4. Filter out false positives and temporal smoothing of detection boxes: For each frame:
	a. Make a heatmap out of all the hot windows found at each scale for this frame (increase a pixel's value every time it is contained within a hot window)
	b. Using an global variable of class Heatmap(), enqueue the heatmap
	c. Compute the average of the last _n_ heatmaps
	d. Threshold the resulting average heatmap to a specified value
	e. Create bounding boxes around the remaining distinct hot areas of the thresholded average heatmap and add them to the original clip frame
	f. Write out the resulting image to the output movie file
	

## Files

Before running step i above, it is advised to extract images from the [Udacity dataset] (https://github.com/udacity/self-driving-car/tree/master/annotations) by using `extra_data.py`. I wrote this code to make it easy to a balanced dataset containing positive and negative cases (ie. 64x64 windows containing a car or not). The negative cases are picked at random on each image in areas outside of the car bouding boxes. Their size is randomized too and resized to 64x64.

Before starting, make sure you have execution rights on all `.py` files (see shell command `chmod` if not)

**Step i:**
 - Run `./feature_extraction.py` (make sure you have execution rights on this file).
 - The feature vectors for positive and negative cases will be saved in `features.p`
 
**Step ii:**
 - Run `./model_training.py`
 - The trained scaler and classifier will be saved in `scaler.p` and `classifier.p` respectively
 
**Steps iii and iv:**
 - Run `./detection.py`
 - The output video clip will be saved as `project_video_result.mp4`
 
Many parameters can be tweaked to improve the pipeline robustness, but be careful about consistency between parameters used when extracting features to train the classifier and those used at detection time. The most common cause of crash is a mismatch in feature vector lengths between what is extracted from each frame of the video clip and what the scaler and classifier expect.

**Other code files:**
 - `class.py` implements a `Heatmap()` class, used to average successive heatmaps
 - `tests.py` is a sandbox that I used to test various parts of the pipeline
 - `project_illustrations_*.py` files were used to produce figures for the project report. They call various functions of the pipeline
 
**Documentation:**
 - `README.md`: This file
 - `project_report.md`: The full project report written for submission to Udacity. It explains design choices, results and discusses ideas for further improvement.
 
 