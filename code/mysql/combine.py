#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import pickle

data_x = []
data_y = []
root_path = ""

dir_or_files = os.listdir(root_path)
for dir_file in dir_or_files:
    dir_file_path = os.path.join(root_path, dir_file)
    data_x0, data_y0 = pickle.load(open(dir_file_path, 'rb'))
    data_x.append(data_x0)
    data_y.append(data_y0)

pickle.dump((data_x, data_y), open('han_type1_data_all', 'wb'))
