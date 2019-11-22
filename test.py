import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
data_root = '/data0/liumengmeng/data/'



shutil.move('/home/liumengmeng/anaconda3','/data0/liumengmeng/')
#shutil.copytree()




# for i in range(3):
#     shutil.copy(data_root +'COCO_train2014_000000000089.png', data_root+'COCO_train2014_000000000089'+ '_' +str(i)+'.png')
#     #shutil.copy(src_path+'/'+ ins_item[:-4] + str(i) +'.jpg', test/coco_background)

#print(os.path.exists('/data0/liumengmeng/data/_a_gt_full/COCO_train2014_000000080168.png'))
# img = cv2.imread('/data0/liumengmeng/data/_a_ori_full/COCO_train2014_000000080168.jpg')
# mask = cv2.imread('/data0/liumengmeng/data/_a_gt_full/COCO_train2014_000000080168.png')
# res_path = '/data0/liumengmeng/data/_a_src_full'  
# b, g, r = cv2.split(mask)
# a = b
# b_pic, g_pic, r_pic = cv2.split(img)
# img_BGRA = cv2.merge((b_pic, g_pic, r_pic, a))
# cv2.imwrite(os.path.join(res_path,'COCO_train2014_000000080168.png'), img_BGRA)