#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
These functions are used to train an machine learning model in order to predict whether a car is
present on a given image. The outcomes of this set of functions are a scaler (to normalize features)
and a trained model (to make predictions).
'''

from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import expon as sp_expon
from scipy.stats import uniform as sp_unif
import pickle
import glob
from feature_extraction import *


def make_train_test_sets(X_positive, X_negative, valid_set = False, test_size = .2, 
    scaler_savefile = None):
    '''
    Takes two feature lists (one for positive cases, one for negative cases) and stacks them into a
    single feature array. Generates a vector of labels with 1s and 0s of corresponding length.
    Fits a scaler for the feature array by feature so that each feature has zero mean and unit 
    variance. Computes the normalized feature array.
    Splits the features array into a train and a test set as per the 'test_size' parameter.
    If 'savefile' != None, saves the scaler to disk with the specified file name.

    Returns: train feature set, test feature set, train labels, test labels, trained scaler.
    '''
    # Stack feature vectors for positive and negative cases:
    X = np.vstack((X_positive, X_negative)).astype(np.float64)
    # Create the target (label) vector:
    y = np.hstack((np.ones(len(X_positive)), np.zeros(len(X_negative))))
    # Normalize the feature vector:
    scaler = RobustScaler().fit(X)
    scaled_X = scaler.transform(X)
    print("Shape:", scaler.scale_.shape)

    if scaler_savefile is not None:
        with open(scaler_savefile, 'wb') as f:
            pickle.dump(scaler, f)

    # Split the data into train and test set:
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = test_size,
        random_state = rand_state)
    if valid_set:
        X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size = .5,
            random_state = rand_state)
        return X_train, X_valid, X_test, y_train, y_valid, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test, scaler


def train_svm_with_cv(X, y, fit_param, n_iter = 20, n_jobs = 3, cv = 5, verbose = 3, 
    clf_savefile = None):
    '''
    Wrapper function that runs a randomized parameter search using cross_validation and fits the 
    best model (uses an SVM classifier).
    - fit_param:    A dict in the form {'parameter': [values]} or {'parameter': distribution}
    - n_iter, n_jobs, cv, verbose: Parameters to RandomizedSearchCV() (see sklearn documentation)
    - clf_savefile: None or the file name to use to dump the best trained classifier
    
    Returns the best trained model, based on accuracy.
    '''
    # Define classifier:
    clf = RandomizedSearchCV(SVC(kernel = 'rbf'), fit_param, 
        n_iter = n_iter, 
        n_jobs = n_jobs, cv = cv, 
        verbose = verbose)
    # Search for the best model and train it
    clf.fit(X, y)

    # Save the classifier to disk if required by user:
    if clf_savefile is not None:
        with open(clf_savefile, 'wb') as f:
            pickle.dump(clf, f)

    return clf


def train_svm_with_known_parameters(X, y, C =  0.05, gamma = 0.01, 
    clf_savefile = None):
    '''
    Wrapper function that fits an SVM classifier to the training data.
    - kernel, C, gamma: SVM parameters (see sklearn documentation)
    - clf_savefile:     None or the file name to use to dump the trained classifier
    
    Returns the trained model.
    '''
    # Define classifier:
    clf = SVC(C = C, gamma = gamma)
    # Train it:
    clf.fit(X, y)

    # Save the classifier to disk if so required:
    if clf_savefile is not None:
        with open(clf_savefile, 'wb') as f:
            pickle.dump(clf, f)

    return clf


def train_mlp_with_known_parameters(X, y, hidden_layer_sizes = (100, 50, 20), 
    clf_savefile = None):
    '''
    Wrapper function that fits an MLP classifier to the training data.
    - hidden_layer_sizes: MLP parameters (see sklearn documentation)
    - clf_savefile:     None or the file name to use to dump the trained classifier
    
    Returns the trained model.
    '''
    # Define classifier:
    clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes)
    # Train it:
    clf.fit(X, y)

    # Save the classifier to disk if so required:
    if clf_savefile is not None:
        with open(clf_savefile, 'wb') as f:
            pickle.dump(clf, f)

    return clf


if __name__ == '__main__':
    with open('features.p', 'rb') as f:
        car_feat = pickle.load(f)
        notcar_feat = pickle.load(f)

    print("Feature vector length:", car_feat[0].shape, notcar_feat[0].shape)


    # Make train and test sets and save fitted scaler
    X_train, X_valid, X_test, \
    y_train, y_valid, y_test, \
    scaler = make_train_test_sets(car_feat, notcar_feat, test_size = .4, valid_set = True,
        scaler_savefile = 'scaler.p')
    
    # Randomized Search parameters

    # Fit model for first round of training
    # clf = train_with_cv(X_train, y_train, parameters, n_iter = 20, n_jobs = 3, cv = 5, verbose = 3, 
    # clf_savefile = 'classifier.p')
    # svm_parameters = {'C': sp_expon(scale = 0.1), 'gamma': sp_expon(scale = 0.12)}

    # clf = train_svm_with_cv(X_train, y_train, svm_parameters, n_iter = 20, n_jobs = 3, 
    #     cv = 3, verbose = 3, clf_savefile = None)
    clf = train_mlp_with_known_parameters(X_train, y_train, 
        hidden_layer_sizes = (1500, 500, 300, 100, 50, 10), 
        clf_savefile = None)
    print('Validation Accuracy of classifier = ', round(clf.score(X_valid, y_valid), 4))

    # Predict on validation set:
    pred_valid = clf.predict(X_valid)
    X_neg = X_valid[pred_valid != y_valid]
    y_neg = y_valid[pred_valid != y_valid]
    print(X_neg.shape)
    print(y_neg.shape)

    # Add negative cases to training_set:
    print(X_train.shape)
    print(y_train.shape)   
    X_train = np.vstack((X_train, X_neg))
    y_train = np.hstack((y_train, y_neg))
    print(X_train.shape)
    print(y_train.shape)

    # Retrain with added negative cases
    # clf = train_svm_with_cv(X_train, y_train, svm_parameters, n_iter = 20, n_jobs = 3, 
    #     cv = 3, verbose = 3, clf_savefile = 'classifier.p')    
    clf = train_mlp_with_known_parameters(X_train, y_train, 
        hidden_layer_sizes = (1500, 500, 300, 100, 50, 10), 
        clf_savefile = 'classifier.p')

    # print('Best parameters:', clf.best_estimator_)
    print('Test Accuracy of classifier = ', round(clf.score(X_test, y_test), 4))
