#!/usr/local/bin/python3
# Copyright 2019-02-24 Powr
#
# All rights reserved
#
# Author: Powr
#
#==================================================================
"""
    This nice script takes a greyscale image (28x28) from MNIST dataset
    and shows the image with the help of matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import sys



if len(sys.argv) is 2 and sys.argv[1] == "-h":
    print("This nice script takes a greyscale image (28x28) from MNIST dataset and shows the image with the help of matplotlib")
    print("To use this file, specify which dataset file and/or image file should be used...")
    print("Train-set = 0, test-set = 1, rotated-train-set = 2, rotated-test-set = 3")
    print("Image-file = 0,1,...")
    print("Usage:\t {} <dataset> <image-number>".format(sys.argv[0]))
    sys.exit()

if len(sys.argv) is not 3:
    print("ERROR - You forgot to specify which dataset file and/or image file should be used...")
    print("Train-set = 0, test-set = 1, rotated-train-set = 2, rotated-test-set = 3")
    print("Image-file = 0,1,...")
    print("Usage:\t {} <dataset> <image-number>".format(sys.argv[0]))
    sys.exit()


dataset = int(sys.argv[1])
image_number = int(sys.argv[2])

if not (0 <= dataset <= 3):
    print("ERROR - Wrong parameter for the dataset...")
    print("Train-set = 0, test-set = 1, rotated-train-set = 2, rotated-test-set = 3")
    print("Image-file = 0,1,...")
    print("Usage:\t {} <dataset> <image-number>".format(sys.argv[0]))
    sys.exit()



dir_name = "mnist_dataset/"
train_name = "mnist_train.csv"
test_name = "mnist_test.csv"
rotated_train_name = "rotated_mnist_train.csv"
rotated_test_name = "rotated_mnist_test.csv"

file_names = [train_name, test_name, rotated_train_name, rotated_test_name]
set_names = ["training", "test", "rotated-training", "rotated-test"]

# full_path will hold path to either train or test file
full_path = dir_name + file_names[dataset]
used_set = set_names[dataset]

print("Reading {} dataset...".format(used_set))


# open dataset file
data_file = open(full_path, 'r')
data_list = data_file.readlines()
data_file.close()

print("Converting image...")

# read image pixels (still as string)
image = data_list[image_number].split(',')
# create 28x28 square matrix, each value holding a greyscale pixel value between 0-255
image_array = np.asfarray(image[1:]).reshape((28,28))

print("Image shows number: {}".format(image[0]))

plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()







