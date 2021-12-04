import os
import cv2
import numpy as np
import random

from utils.func import *
from pin.pinImage import PinImage


if __name__ == '__main__':
    pinimage = PinImage()
    
    data_root = '/data1/liumengmeng/our_data_CG/'
    n = 'FirstBatch'
    src_path = data_root + '_obj_all/'
    dst_path = data_root + '_bg_all/'
    center_num = 4480
    paths = makepath(n, data_root)
    makefolder(paths)
    pinimage.batch(src_path, dst_path, paths[0], paths[1], center_num, augmented_flag=True, bg_repeat_flag=False)

