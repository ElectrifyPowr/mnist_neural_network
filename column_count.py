#!/usr/local/bin/python3
# Copyright 2019-03-01 Powr
#
# All rights reserved
#
# Author: Powr
#
#==================================================================
"""

    To check whether you have the correct number of columns in a dataset in a specific row
    this script takes 2 arguments:
        1. the file_name
        2. the row

    This is helpful when adding your own images.

"""

import sys


csv_file = sys.argv[1] # "self_made_number1_edit.csv"
row = int(sys.argv[2])

with open(csv_file, 'r') as f:
    data = f.readlines()

values = data[row].split(',')

print("Columns: {}".format(len(values)))



#if __name__ == '__main__':







