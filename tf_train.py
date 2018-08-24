import tensorflow as tf
import numpy as np
import cv2
import sys

filenames = "BSDS500/data/_Preprocess_output.csv"

filename_queue = tf.train.string_input_producer([filenames])
reader = tf.TextLineReader()
record_defaults = [[1], [1], [1], [1]]
key, value = reader.read(filename_queue)
col1, col2, col3, col4 = tf.decode_csv(value, record_defaults=record_defaults)
data = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    _data = sess.run(data)
    print(_data)

