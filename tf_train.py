import tensorflow as tf
import numpy as np
import csv

with open("BSDS500/data/Preprocess_output.csv", newline="") as csvfile:
    data = csv.DictReader(csvfile)
    data = list(data)
