import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
data_root = '/data0/liumengmeng/data/'

#test-----------------------------------------------------------------
fg = cv2.imread(data_root+"_a_src_full/COCO_train2014_000000000332.png",-1)
bg = cv2.imread(data_root+"_a_dst/bing_bg_1_0105.jpg")
ins = cv2.imread(data_root+"_a_ins/COCO_train2014_000000000853.png",-1)
gt_4667 = cv2.imread(data_root+"_a_gt_full/COCO_train2014_000000000110_3.png",-1)

#----------------cv2.imshow()-------------------------
# # cv2.imshow('image',rotate(fg))
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()   #cv2.destroyWindow(wname)

# with open('test/test.txt','a') as f:
#     f.write('\n'+'cococococcocco')
# with open('test/test.txt') as f:
#     a = [line.rstrip() for line in f]
# print(a)

a = [1,2,3,4]
b = [1,3,4,6,7,8,]
for i in a:
    if i not in b:
        print(i)

#shutil.move('/home/liumengmeng/anaconda3','/data0/liumengmeng/')
#shutil.copytree()
#shutil.copy('/home/liumengmeng/DSS/old/results_50epoch_1e4/run-2/models/final.pth','/home/liumengmeng/DSS/DSS-pytorch/weights')



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