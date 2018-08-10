import scipy.io as sio
import numpy as np
from os import listdir
import tensorflow as tf
import cv2
import os

target_dir = "BSDS500/data/"
train_img = listdir(target_dir + "images/train/")

for _train_img in train_img:

    ori_img = cv2.imread(target_dir + "images/train/" + _train_img)
    # read images from the train images directory
    file_ext = os.path.splitext(os.path.basename(_train_img))
    train_gndTrue = sio.loadmat(target_dir + "groundTruth/train/" + file_ext[0] + ".mat")
    # read the corresponding ground truth file from the train ground truth directory.

    biliteral_img = cv2.bilateralFilter(ori_img, 5, 60, 60)
    # bilateral smooth the original image to erase less important detail

    edges_map = np.zeros((ori_img.shape[0], ori_img.shape[1]), np.uint8)
    # Create an empty image with the same reso from the ori image.

    for i in train_gndTrue["groundTruth"][0]:

        edges_map = edges_map + i[0][0][1].astype(np.uint8)
        edges_map = cv2.equalizeHist(edges_map)

    norm_img = cv2.normalize(edges_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow("Original image", ori_img)
    cv2.imshow("bill", biliteral_img)
    cv2.imshow('black', edges_map)
    np.set_printoptions(formatter={'float': lambda norm_img: "{0:0.3f}".format(norm_img)})
    print(norm_img)
    cv2.waitKey(0)
