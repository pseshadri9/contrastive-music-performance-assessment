import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io

SEGMENT = '2'

# read middle school data
BAND = 'middle'
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = '../dat/' + file_name + '.dill'
else:
    data_path = '../dat/' + file_name + '_3.dill'
mid_data = dill.load(open(data_path, 'rb'))

# read symphonic band data
BAND = 'symphonic'
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = '../dat/' + file_name + '.dill'
else:
    data_path = '../dat/' + file_name + '_3.dill'
sym_data = dill.load(open(data_path, 'rb'))

# combine data
com_data = mid_data + sym_data

# create write file name and write data
BAND = 'combined'
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    with open('../dat/' + file_name + '.dill', 'wb') as f:
        dill.dump(com_data, f)
    scipy.io.savemat('../dat/' + file_name + '.mat', mdict = {'perf_data': com_data})
else:
    with open('../dat/' + file_name + '_3.dill', 'wb') as f:
        dill.dump(com_data, f)

