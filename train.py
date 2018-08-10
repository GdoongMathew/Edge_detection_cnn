import scipy.io as sio
import numpy as np
from os import listdir
import tensorflow as tf
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
    crop_ori_img = None
    crop_gnd_truth = None

    for h in range(ori_img.shape[1] - crop_res):
        for w in range(ori_img.shape[0] - crop_res):
            if first:
                crop_ori_img = np.array([ori_img[w: w + crop_res, h: h + crop_res]])
                crop_gnd_truth = np.array([norm_img[w: w + crop_res, h: h + crop_res]])
                first = not first
            else:

                crop_ori_img = np.append(np.array([ori_img[w: w + crop_res, h: h + crop_res]]), crop_ori_img, axis=0)
                crop_gnd_truth = np.append(np.array([norm_img[w: w + crop_res, h: h + crop_res]]), crop_gnd_truth, axis=0)

    # Need to find all zeros images and reduce the number of them to make the ratio of negative/positive near 2:1

    #cv2.destroyAllWindows()
    total_image.append(biliteral_img)
    total_edge_map.append(edges_map)


