#!/usr/local/bin/python3
# Copyright 2019-02-25 Powr
#
# All rights reserved
#
# Author: Powr
#
#==================================================================
"""

    Generates 2 more images for each original image of dataset, namely
        - image rotated to the left by 10 degrees
        - image rotated to the right by 10 degrees

"""

import numpy as np
import matplotlib.pyplot as plt
# scipy.ndimage for rotating image arrays
import scipy.ndimage
from itertools import chain # converting 2d np_arrays to flat lists
import random



img_rotation = 10.0

dir_name = "mnist_dataset/"
train_name = "mnist_train.csv"
test_name = "mnist_test.csv"
rotated_train_name = "rotated_mnist_train.csv"
rotated_test_name = "rotated_mnist_test.csv"

file_names = [train_name, test_name]
rotated_file_names = [rotated_train_name, rotated_test_name]


for l in range(len(file_names)):
    print("Rotating all images of {} by {}°".format(file_names[l], img_rotation))

    # full_path will hold path to either train or test file
    full_path = dir_name + file_names[l]
    #full_path = dir_name + test_name

    # open dataset file
    data_file = open(full_path, 'r')
    data_list = data_file.readlines()
    data_file.close()


    all_images = []


    for i in range(len(data_list)):
        # read image pixels (still as string)
        image = data_list[i].split(',')
        # scale input to range 0.01 to 1.00
        scaled_input = ((np.asfarray(image[1:]) / 255.0 * 0.99) + 0.01).reshape(28,28)

        label = int(image[0])

        # create rotated variations
        # rotated anticlockwise by 10 degrees
        inputs_plus10_img = scipy.ndimage.rotate(scaled_input, img_rotation,
                                                 cval=0.01, order=1, reshape=False)
        inputs_plus10_img = (inputs_plus10_img - 0.01) * (255*0.99)
        inputs_plus10_img = inputs_plus10_img.astype(int)
        # rotated clockwise by 10 degrees
        inputs_minus10_img = scipy.ndimage.rotate(scaled_input, -img_rotation,
                                                  cval=0.01, order=1, reshape=False)
        inputs_minus10_img = (inputs_minus10_img - 0.01) * (255*0.99)
        inputs_minus10_img = inputs_minus10_img.astype(int)

        # convert numpy 2d-array back to flat list & add label at first index to list
        in_plus10 = [label] + list(chain.from_iterable(inputs_plus10_img))
        in_minu10 = [label] + list(chain.from_iterable(inputs_minus10_img))
        #original_img = [label] + list(chain.from_iterable(scaled_input))

        # add original & rotated images to all images
        original_img = list(map(int, image))
        all_images.append(original_img)
        all_images.append(in_plus10)
        all_images.append(in_minu10)

    print("Shuffling now all rotated & original images (size: {})".format(len(all_images)))
    # shuffling list so that each number doesn't appear 3 times each time in a row
    random.shuffle(all_images)


    print("Saving now all images to file: {}".format(rotated_file_names[l]))
    # save all images to file
    full_path = dir_name + rotated_file_names[l]
    with open(full_path, 'w') as f:
        for i, val in enumerate(all_images):
            for k, value in enumerate(all_images[i]):
                separator = ","
                if k == len(all_images[i])-1:
                    separator = ""
                f.write("%s%s" % (value, separator))
            f.write("\n") # write empty line











