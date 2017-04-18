##Vehicle Detection and Tracking
###Udacity Self-Driving Car Nanodegree - Project 5

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run this pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1a]: ./output_images/car_image.png
[image1b]: ./output_images/feature_extraction.png
[image1c]: ./output_images/feature_plot.png
[image2]: ./output_images/udacity_images.png
[image3]: ./output_images/sliding_window.png
[image4]: ./output_images/detection_window.png 
[image5]: ./output_images/heatmaps.png
[image6]: ./output_images/bounding_boxes.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function that extracts HOG features is `get_hog_features()`, on line 110 of `feature_extraction.py`. In addition to HOG features, I also extracted color features by using spatial binning and color histograms. These are extracted in functions `bin_spatial()` (line 99) and `color_hist()` (line 80) respectively.

These three functions are called by `extract_features_from_images()` (line 139). They are all adapted from the code provided in the project lectures. The main difference is that I included the ability to use different color spaces for HOG and color features.
`extract_features_from_images()`takes a list of image filenames or images as Numpy arrays and applies the 3 previously mentioned functions to it to generate a feature vector. 

####2. Explain how you settled on your final choice of HOG parameters.

I tried different sets of parameters for each of these features, initially on a subset of the data to speed up exploratory analysis. My explorations led me to the following conclusions:

- HLS color space offers the best results for color features, all 3 channels
- YCrCb color space offers the best results for HOG features, all channels
- Spatial size: 16x16 does not degrade performance compared to 32x32 while creating a smaller vector, so it is a better choice
- Histogram bins: 64 bins offer significantly better performance than 32 or less.
- HOG orientations: Less than 9 degrade performance. More does not help but significantly increases the vector size
- HOG pixels per cell: 16 seems to offer the best performance while also reducing the vector size
- Cells per block: 2 and 4 offer similar performance, but 4 effectively means only 1 block per 64x64 window, so no block normalization. While this seemed to work well on the training and test images, I suspect this would not generalize as well with the clip images. I therefore settled for 2.

The figures below show how each of these functions turn an image into feature vectors:
![alt_text][image1a]
![alt_text][image1b]
![alt_text][image1c]

The final plot show why normalizing the data will be crucial when training a classifier: Our feature variables have very different scales.

**NOTE:** As we will see later in the Video section of this report, these optimal parameters turn out not to be so optimal when appied to a video clip. I actually had to turn off spatial binning and color histograms.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before running a classifier, I normalized the data using RobustScaler (from sklearn) which is more resistant to outliers than StandardScaler.

Initial trials were run with the standard data set and a Linear SVM. To discover the best parameters, I used randomized search rather than grid search, because I found it much more efficient in finding good parameters in a given time. 

Despite getting over 99% test accuracy, I was getting too many false positives on the movie clip that I was not able to get rid of despite tweaking search window scales, heatmap thresholding and averaging frames. I therefore decided to use the Udacity dataset, which contains around 62,000 bounding boxes around cars. To get a balanced set, I also wrote code to generate non-car boxes of random sizes and at random locations.

The code for all of this is in file `extra_data.py`.

Here are some example images extacted from the Udacity dataset:
![alt_text][image2]

As you can seen, this dataset is quite hard. I was annotated by humans who are able to infer information from context, therefore some of the boxes labelled as cars are difficult to recognize when seen in isolation. Additionally, the original bounding boxes can be of any shape, and the extraction process resizes them to 64x64, thus inducing distortion, sometimes severe.

I separated the data set into a training, a validation and a test set (60 / 20 / 20).

However I discovered that with this harder data, a Linear SVM was no longer powerful enough as I could not get above 90% test accuracy. I tried a Gaussian SVM but the training time became unrealistically long. I therefore decided to try a Multi-layer Perceptron neural network, which worked much better and trained much faster.

I then predicted results on the validation set and isolated wrong predictions, which I then added back to my training set. I ran a second training pass with these additional observations (hard negative mining). This second pass helped gain a further 1% in accuracy and exceed 96%.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The detection part of the pipeline is in file `detection.py`. The function doing the heavy lifting is `find_cars()` (line 20). It takes an images, converts it to the right color spaces (in my case, HLS and YCrCb) and resizes it depending on the window scale being used. It then applies `get_hog_feature()` to the whole image (or rather, the region of interest, ie. from 54% of the vertical axis).

Then it slides a window accross the image and calls the other feature extraction functions to create a feature vector for that particular image patch. 

Finally, it runs the classifier on this feature vector to make a prediction on whether or not there is a car in that patch. If there is, the window coordinates are appended to a list of 'hot' windows.

The function `scan_at_scales()` (line 136) runs `find_cars()` at different scales on a specified region of interest of the image. It extends the list of hot windows with those found at different scales.

The image below illustrates how i used different scales for different parts of the image (see section below for an explanation). I used random window colors to help distinguish between windows.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As previously explained, I started optimizing the feature vector size and classifier parameters using a subset of the standard data, then moved on to the full data set.
As the results were still unsatisfactory, I further expanded the data by including the Udacity dataset and switching to a neural network predictor, which helped a lot with sensitivity. Specificity was addressed by other methods (see below).

Using the same image as in the previous section, here's what the classifier identified as cars using a probability threshold of 0.999:

![alt text][image4]

As you can see, we sill have some false detections. We will address this in the next chapter.

To improve execution speed, I restricted window scales to only 3 values (0.70, 1.0 and 2.0) and the vertical search area to 54%-75% of the image at the smallest scale, then 54%-100% at the other two.
The reasoning behind this is that smaller instances of cars will appear closer to the horizon, so I don't need to run the classifier at the smallest scale over the whole region of interest.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_result.mp4). I am still getting a few breaks in detection, but overall cars are correctly identitified most of the time.
One important notion to consider is that even the best classifier will not achieve 100% accuracy. While this might be good enough on individual images, the project video contains over 1200 frames. On each frame, I am running the classifier nearly 600 times. So over the course of the full video, we would inevitably get several thousands of wrong predictions, either false positives or false negatives. Hence the importance of having a process to filter out wrong predictions further down the pipeline.

One thing that I noticed is that using only HOG features (ie. removing spatial and histogram features) reduced the prediction accuracy on still images, but generated significantly less false positives on the video while retaining enough positive predictions. I therefore decided to use HOG features only, which has the added benefit of simplifying the model slightly.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Positive detections that exceed a prediction certainty threshold are used to create a cumulative heatmap for each movie frame (`make_heatmap_from_frame()`, line 191 of file `detection.py`). Every time a positive detection is found, some heat is added to the heatmap (`add_heat()`, line 164). 

Successive heatmaps are then averaged within the custom `Heatmap()` class (`weighted_average()` on line 63 of file `classes.py`). 

This average heatmap is then thresholded to retain only the highest values (`apply_threshold()`, line 180 of file `detection.py`).

The thresholded average heatmap is then passed to `draw_labeled_bboxes()` on line 222 of `detection.py`. This function generates labels for distinct hot areas of the heatmap and draws bounding boxes around each of these areas.

There are three parameters I used to tune false positive filtering:

- `prob_thresh`: Probability threshold for a window being considered as a positive case. 
Decreasing this value will increase the classifier's sensitivity but decrease specificity as more uncertain detections will be considered as positive. The good thing about the MLP classifier is that it gives clearly contrasted probability values so I was able to use a high value to filter out most false detections while retaining enough true positives. I used 0.9995, which eliminates the vast majority of false positives. Beyond this value, I am starting to get too many breaks in detection.

- `hm_thresh`: Value threshold for a heatmap patch to be considered as a valid detection.
Increasing this value means that more detection windows need to be classified as positive over a given patch of the image for it to be considered as a valid car detection. I used 0.6 (values are averaged over several frames so they are not necessarily integers).

- `q_length`: Number of successive frames to average heatmaps over.
Heatmap values are averaged over this many frames. Increasing this value yields to more stable detection boxes and less positive detections (as a flickering detection might not have enough weight to exceed the heatmap threshold value), but less reactivity to new vehicles entering the field of view for instance, and longer persistence of false detections. I used 15.


### Here are the original images, raw heatmaps, average heatmaps (on 3 frames in this examples -- the real code uses 15) and thresholded average heatmaps for 6 frames:
![alt text][image5]


### Here are the resulting bounding boxes:
![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, the main issue I had with this project was trying to get rid of false positives. This is what let me to increasing the amount of training data, thus forcing me to use a different classifier algorithm etc. I could certainly have taken a much simpler approach (many fellow students did) but I hope that all this work results in a more robust pipeline that _should_ generalize better to real-life situations. Once I started using an MLP, false positives were less of an issue because I could basically set the detection probability threshold as high as I needed. I am stil getting some though, and there is a trade-off with detection breaks.

There is still a slight issue around multiple detections: Whenever several cars (or elements perceived as such) are present on the frame, their respective weights in the average heatmap is reduced by a factor equal to the number of cars. This means that occasionally, the weight might be lower than the heatmap threshold and the cars not marked as detected. Perhaps an even more accurate classifier would allow us to further increase the detection threshold, thus getting rid of all false positives, which would then let us decrease the heatmap threshold further. This would certainly help with this issue. This is probably more a matter of tweaking and of time available than a general redesign of the pipeline, though.

I am not totally satisfied by the output on the project video to be honnest. I wish I had been able to spend more time optimizing the MLP's architecture and parameters to improve accuracay further, which would give me more breathing space to tweak sensitivity vs. specificity with the heatmap parameters. This is something I intend to keep working on after submitting, and I might re-submit if I achieve significant improvements.

An idea to improve robustness would be to use two (or more) cameras. With the offset between them, they would each provide a slightly different image, thus increasing the probability for successful detections. With an MLP classifier, we showed that adjusting the probability threshold is enough to get rid of almost all false positives.

Beside robustness, one major point that would need to be addressed before this pipeline can be used in real-life applications is execution speed. Currently, my pipeline is able to process frames at a rate of around 2 seconds per frame. For real-time execution, this would need to be improved by at least one order of magnitude. Maybe translating the code to C++ would help. Although the OpenCV library is already written in that language, it would make array manipulation faster but it is doubtful that the gains would be sufficient. OpenCV offers an alternative to skimage's `hog()` function that is faster but is poorly documented and (seemingly) doesn't offer the option to return a 5d array instead of a linear vector, which is required within my `find_cars()` function.

Another way to improve speed and maybe reliability would be to train the model with 64x64 windows showing cars at many different scales (ie. some of them zoomed in on small portions of cars, while others would show mostly empty space with a small car in the middle). With enough such data (possibly obtained through data augmentation), only one sliding window pass might be enough to ensure good detection quality and would be significantly faster. This seems like a fairly easy thing to implement using the Udacity dataset.

I also saw some attempts at using CNNs but it was unclear to me whether they were a better or faster alternative. This could be an interesting development.

