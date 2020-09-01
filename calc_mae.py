import os
from PIL import Image
import numpy as np
import math
from utils.func import *

#---------------------------all datasets------------------------------------
namelist = ['DUTS','DUT-OMRON','ECSSD','PASCAL-S','SOD']

def save_mae(save_folder,mae_id_folder):
    '''
    save_folder: the folder to save the high mae pics --img /gt /map
    mae_id_folder : the folder that saves all high mae id
    '''
    num = 0
    for i in namelist:
        num +=1
        paths = []
        img_root = '/data1/liumengmeng/dataset/' + i + '/imgs'
        gt_root = '/data1/liumengmeng/dataset/' + i + '/gt'
        test_mae_root = os.path.join(save_folder, i + '/map')
        test_img_root = os.path.join(save_folder, i + '/img')
        test_gt_root = os.path.join(save_folder, i + '/gt')
        paths.append(test_mae_root)
        paths.append(test_img_root)
        paths.append(test_gt_root)
        makefolder(paths)
        map_root = '/data1/liumengmeng/DSS/DSS-results/test-'+ str(num) + '/test_map'
        mae_id_path = os.path.join(mae_id_folder, i + '.txt')
        copy_by_idtxt(mae_id_path, map_root, paths[0], '.png')
        copy_by_idtxt(mae_id_path, img_root, paths[1], '.jpg')
        copy_by_idtxt(mae_id_path, gt_root, paths[2], '.png')
        


# save_folder = '/data1/liumengmeng/CG4_test'
# mae_id_folder = '/data1/liumengmeng/CG4_id_mae'
# save_mae(save_folder,mae_id_folder)

#----------------------pick singel datasets----------------------------------------

dataset = 'HKU-IS' #拷贝rbg和gt的数据集
i = 'HKU-IS' #测试结果的名字/txt/...
num = 0  #DSS-results中的test的folder number
save_folder_name = 'HKU-IS-better'
save_folder = '/data1/liumengmeng/CG4_test'
mae_id_folder = '/data1/liumengmeng/CG4_id_mae'
# ------------
# map_root = '/data1/liumengmeng/DSS/DSS-results/test-'+ str(num) + '/test_map'
map_root = '/data1/liumengmeng/DSS/DSS-results-n2/test-11/test_map'
mae_id_path = os.path.join(mae_id_folder, i + '.txt')
# mae_id_path = '/data1/liumengmeng/DSS/DSS-results/test-'+ str(num) + '/mae_id.txt'
#-------------
paths = []
img_root = '/data1/liumengmeng/dataset/' + dataset + '/imgs'
gt_root = '/data1/liumengmeng/dataset/' + dataset + '/gt'
test_mae_root = os.path.join(save_folder, save_folder_name + '/map')
test_img_root = os.path.join(save_folder, save_folder_name + '/img')
test_gt_root = os.path.join(save_folder, save_folder_name + '/gt')
paths.append(test_mae_root)
paths.append(test_img_root)
paths.append(test_gt_root)
makefolder(paths)

copy_by_idtxt(mae_id_path, map_root, paths[0], '.png')
copy_by_idtxt(mae_id_path, img_root, paths[1], '.png')
copy_by_idtxt(mae_id_path, gt_root, paths[2], '.png')