#!/usr/bin/python

import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

global svc
global scaler

#test_data_dir = 'test_images'
#test_data_dir = 'training_data/non-vehicles/GTI'
test_data_dir = 'training_data/vehicles/KITTI_extracted'
classifier_data_dir = 'training_data'
vehicle_data = os.path.join(classifier_data_dir, 'vehicles/KITTI_extracted')
non_vehicle_data = os.path.join(classifier_data_dir, 'non-vehicles/GTI')

classifier_labels = {1: 'car', 0: 'not-car'}

def add_heat(heatmap, box):
    heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap >= threshold] = 1
    # Return thresholded map
    return heatmap

def draw_boxes(img, bbox, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def extract_features(bgr_img, viz = False):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hog_features, feature_img = hog(gray, orientations = 9, pixels_per_cell=(8, 8), 
            cells_per_block = (2, 2), visualise = True, feature_vector = True, block_norm="L2-Hys")
    if viz:
        plt.subplot(2,1,1)
        plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        plt.subplot(2,1,2)
        plt.imshow(feature_img)
        plt.show()
    return hog_features.ravel()

def train_model(car_features, non_car_features):
    global svc
    global scaler
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return (svc, scaler)

def sliding_window(img, x_start_stop = [None, None], y_start_stop = [None, None],\
        xy_window = (64, 64), xy_overlap = (0.5, 0.5)):
    window_list = []
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def testPipeline(bgr_img):
    global svc
    global scaler

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    candidate_windows = []
    #window_sizes = [64, 128, 256]
    window_sizes = [64]

    for window in window_sizes:
        candidate_windows = candidate_windows + sliding_window(rgb_img, x_start_stop=[None, None],
                y_start_stop=[rgb_img.shape[0] // 2, None], 
                   xy_window=(window, window), xy_overlap=(0.5, 0.5))

    hot_windows = []
    for window in candidate_windows:
        #print("Window: ", window)
        test_img = cv2.resize(bgr_img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        features = extract_features(test_img)
        scaled_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = svc.predict(scaled_features)
        if prediction == 1:
            #print("Hot window: ", window)
            hot_windows.append(window)

    bbox_img = rgb_img
    heatmap = np.zeros_like(rgb_img[:,:,0]).astype(np.float)
    for window in hot_windows:
        #print(bbox_img.shape)
        #bbox_img = draw_boxes(bbox_img, window)
        heatmap = add_heat(heatmap, window)
        heatmap = apply_threshold(heatmap, 2)
        labels = label(heatmap)
        bbox_img = draw_labeled_bboxes(rgb_img, labels)

    return bbox_img
    
if __name__ == '__main__':
    global svc
    global scaler

    vehicle_features = []
    non_vehicle_features = []

    print("Loading features..")
    for img_name in os.listdir(vehicle_data):
        bgr_img = cv2.imread(os.path.join(vehicle_data, img_name))
        if bgr_img is not None:
            vehicle_features.append(extract_features(bgr_img))

    print("Loaded vehicle features {}".format(len(vehicle_features)))
    for img_name in os.listdir(non_vehicle_data):
        bgr_img = cv2.imread(os.path.join(non_vehicle_data, img_name))
        if bgr_img is not None:
            non_vehicle_features.append(extract_features(bgr_img))

    print("Loaded non-vehicle features {}".format(len(non_vehicle_features)))

    svc, scaler = train_model(vehicle_features, non_vehicle_features)


    #for img_name in os.listdir(test_data_dir):
    #    bgr_img = cv2.imread(os.path.join(test_data_dir, img_name))
    #    if bgr_img is not None:
    #        print('Opened: {}'.format(img_name))
    #        #print(bgr_img.shape)
    #        #extract_features(bgr_img, True)
    #        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    #        windows = sliding_window(rgb_img, x_start_stop=[None, None], y_start_stop=[None, None], 
    #                   xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    #        break;

    input_video = os.path.join('.', 'project_video.mp4')
    output_video = os.path.join('.', 'project_video_output.mp4')
    #input_video = os.path.join('.', 'test_video.mp4')
    #output_video = os.path.join('.', 'test_video_output.mp4')
    clip = VideoFileClip(input_video)
    output_clip = clip.fl_image(testPipeline)
    output_clip.write_videofile(output_video, audio=False)
