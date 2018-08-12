import scipy.io as sio
import numpy as np
from os import listdir
import tensorflow as tf
import random
import cv2
import os

target_dir = "BSDS500/data/"
train_img = listdir(target_dir + "images/train/")
gnd_data = listdir(target_dir + "groundTruth/train")

total_edge_map = [None] * len(train_img)
# Create an empty list to store all edge images
total_image = [None] * len(train_img)

for _train_img in train_img:

    print(_train_img)

    ori_img = cv2.imread(target_dir + "images/train/" + _train_img)
    # read images from the train images directory
    file_ext = os.path.splitext(os.path.basename(_train_img))

    if str(file_ext[0] + ".mat") in gnd_data:
        train_gndTrue = sio.loadmat(target_dir + "groundTruth/train/" + file_ext[0] + ".mat")
    else:
        print(file_ext[0])
        continue
    # read the corresponding ground truth file from the train ground truth directory.

    biliteral_img = cv2.bilateralFilter(ori_img, 5, 60, 60)
    # bilateral smooth the original image to erase less important detail

    edges_map = np.zeros((ori_img.shape[0], ori_img.shape[1]), np.uint8)
    # Create an empty image with the same reso from the ori image.

    for i in train_gndTrue["groundTruth"][0]:

        edges_map = edges_map + i[0][0][1].astype(np.uint8)
        edges_map = cv2.equalizeHist(edges_map)

    norm_img = cv2.normalize(edges_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Normalize pixel values into 0 and 1
    """
    cv2.imshow("Original image" + _train_img, ori_img)
    cv2.imshow("bill" + _train_img, biliteral_img)
    cv2.imshow('black' + _train_img, edges_map)
    """
    crop_res = 23  # Cropped image resolution
    """
    Crop the image into multiple images and store them into one big array
    """
    first = True
    crop_ori_img = None      # An array to store all cropped original images.
    crop_gnd_truth = None    # An array to store all cropped edge map images.
    crop_img_index = list()  # a list that stores the locations of all cropped edge map images.

    num_total_cropped = (ori_img.shape[1] - crop_res) * (ori_img.shape[0] - crop_res)

    print(ori_img.shape)
    print("Total cropped images:" + str(num_total_cropped))

    for h in range(ori_img.shape[0] - crop_res):
        for w in range(ori_img.shape[1] - crop_res):
            if first:

                # Copy the first crop image into crop_ori_img/ crop_gnd_truth if this's the first crop.
                crop_ori_img = np.array([ori_img[h: h + crop_res, w: w + crop_res]])
                crop_gnd_truth = np.array([norm_img[h: h + crop_res, w: w + crop_res]])
                crop_img_index.append(w + h * (ori_img.shape[0] - crop_res))
                first = not first
            else:
                """
                Reduce the amount of zeros matrix to make the ration of negative/ positive near 2:1
                """
                if np.count_nonzero(norm_img[h: h + crop_res, w: w + crop_res]):
                    # Not a zero matrix, then store them in an array
                    crop_ori_img = np.append(np.array([ori_img[h: h + crop_res, w: w + crop_res]]), crop_ori_img, axis=0)
                    crop_gnd_truth = np.append(np.array([norm_img[h: h + crop_res, w: w + crop_res]]), crop_gnd_truth, axis=0)
                    crop_img_index.append([h, w])
                    cv2.imshow("crop ori img", ori_img[h: h + crop_res, w: w + crop_res])
            cv2.waitKey(15)

    zeros_mat = list(set(list(range(num_total_cropped))) - set(crop_img_index))    # Store index of all zeros crop images 
    zeros_mat = random.shuffle(zeros_mat)          # Shuffle the list
    num_neg_img = int(len(crop_img_index) / 2)  # Only need half the amount of positive images.
    for i in range(num_neg_img):
        """
        iterate through the zeros_mat. Find the location of no edge image and its corresponding original image.
        """
        crop_w = int(zeros_mat[i] % (ori_img.shape[1] - crop_res))
        crop_h = int(zeros_mat[i] / (ori_img.shape[1] - crop_res))
        tmp_ori_img = ori_img[crop_h: crop_h + crop_res, crop_w: crop_w + crop_res]
        tmp_norm_img = norm_img[crop_h: crop_h + crop_res, crop_w: crop_w + crop_res]
        crop_ori_img = np.append(np.array(tmp_ori_img), crop_ori_img, axis=0)
        crop_gnd_truth = np.append(np.array(tmp_norm_img), crop_ori_img, axis=0)

    print(crop_ori_img.shape)
    print(crop_gnd_truth.shape)
    #cv2.destroyAllWindows()
    total_image.append(biliteral_img)
    total_edge_map.append(edges_map)
