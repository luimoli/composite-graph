import os
import math
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
import math
from skimage import transform,data
# import torch
data_root = '/data0/liumengmeng/datasets/data_CG/'

#test-----------------------------------------------------------------
fg = cv2.imread(data_root+"_a_src_full/COCO_train2014_000000011147.png",-1)
bg = cv2.imread(data_root+"_a_dst/opencountry_natu979.jpg")
# ins = cv2.imread(data_root+"_a_ins/COCO_train2014_000000000853.png",-1)
gt_4667 = cv2.imread(data_root+"tmp/_a_gt_full/COCO_train2014_000000000110_3.png",-1)
#----------------------------------------------------------------------------------
# fg1 = cv2.resize(fg,(600,600))
# bg1 = cv2.resize(bg,None,fx=1.5,fy=1.5, interpolation = cv2.INTER_CUBIC)



# img = bg.copy()
# equ = cv2.equalizeHist(img)
# res = np.hstack((img,equ)) #stacking images side-by-side
# cv2.imwrite('res.png',res)

# data=np.array(hist,dtype='uint8')


# # a=np.random.random((5,4,3))
# a = np.random.randint(0,255,(10,10,3))
# c = np.ones((5,5,3))

# # print(a.astype(np.uint8).dtype)
# # print(type(a.astype(np.uint8)[0][0][0]))

# a = np.random.randint(1,4,(2,2,3))
# print(a == 2)
# print(np.argwhere(a == 3))


# ==========================================================================
# with open('test/test.txt','a') as f:
#     f.write('\n'+'cococococcocco')
# with open('test/test.txt') as f:
#     a = [line.rstrip() for line in f]
# print(a)

# with open("a1.txt","r",encoding="utf-8") as f:
#     data = json.loads(f.readline())
#     print(data["COCO_conver"][0]["name"])

# path = open("./test.txt", 'w')
# for i in range(10):
#     print(i,file=path)
# ===========================================================================



# cg_root = '/data0/liumengmeng/CG/'
# with open('/data0/liumengmeng/CG/id/test_id.txt') as f:
#     a = [line.rstrip() for line in f]
# for i in a:
#     shutil.copy(cg_root+'gt/'+ i + '.png', cg_root+'gt_test/')







# for i in range(3):
#     shutil.copy(data_root +'COCO_train2014_000000000089.png', data_root+'COCO_train2014_000000000089'+ '_' +str(i)+'.png')
#     #shutil.copy(src_path+'/'+ ins_item[:-4] + str(i) +'.jpg', test/coco_background)
# 因为某张图的格式是PNG，不是png，所以读入错误，故单独生成某张图片的mask
#print(os.path.exists('/data0/liumengmeng/data/_a_gt_full/COCO_train2014_000000080168.png'))
# img = cv2.imread('/data0/liumengmeng/data/_a_ori_full/COCO_train2014_000000080168.jpg')
# mask = cv2.imread('/data0/liumengmeng/data/_a_gt_full/COCO_train2014_000000080168.png')
# res_path = '/data0/liumengmeng/data/_a_src_full'  
# b, g, r = cv2.split(mask)
# a = b
# b_pic, g_pic, r_pic = cv2.split(img)
# img_BGRA = cv2.merge((b_pic, g_pic, r_pic, a))
# cv2.imwrite(os.path.join(res_path,'COCO_train2014_000000080168.png'), img_BGRA)