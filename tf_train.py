import tensorflow as tf
import numpy as np
import cv2
import csv
import sys

maxInt = sys.maxsize
decrement = True


while decrement:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

# file location
filenames = "BSDS500/data/Preprocess_output.csv"

# Create empty list for later storing all images and ground trouth images.
crop_resolution = []    # resolution of cropped imagee
crop_gnd_truth = []     # ground truth images
crop_img_index = []     # the index where the cropped image originally located.
img_name = []           # original image names without file extension.
with open(filenames, newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        crop_resolution.append(int(row["crop_resolution"]))
        crop_gnd_truth.append(int(row["crop_gnd_truth"]))
        crop_img_index.append(int(row["crop_img_index"]))
        img_name.append(int(row["img_name"]))


for name in img_name:
    ori_img = cv2.imread("images/train/" + str(name) + ".jpg")


crop_gnd_truth = np.asarray(crop_gnd_truth)
print(crop_gnd_truth)
#tensor_gnd_truth = tf.convert_to_tensor(crop_gnd_truth)