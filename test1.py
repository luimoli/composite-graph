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
# import torch as t
# from torchvision.transforms import ToPILImage
# import torch.nn.functional as F
from tqdm import tqdm
from utils.func import *

data_root = '/data0/liumengmeng/data/'


# A=np.array([[[1,2,3,4,5],[2,3,4,5,6]],
#             [[3,4,5,6,7],[4,5,6,7,8]]])
# print(A.shape)
# A= t.from_numpy(A)
# print(A.shape)
# print(A.dim())
# num_a = len(np.argwhere(A > 0))
# print(num_a)

# c = np.random.random((1,4,3))
# c = np.array([[[1,1,1,1,0,1],
#              [1,0,1,0,1,1],
#              [1,1,1,1,1,1]]])
# # c1 = F.softmax(c, dim=0)
# c = c.astype(np.uint8)
# num_a = len(np.argwhere(c == 0))
# print(num_a)
# c= t.from_numpy(c)
# print(c)
# img = ToPILImage()(c)
# img.save('t1.png')

# obj = np.zeros((200,500,1))#创建一个三通道的图
# obj = obj.astype(np.uint8)#改变数据类型为uint8
# cv2.imwrite('t2.jpg', obj)

# for i in range(3):
#     try:
#         with open('./name.txt') as f:
#             a = [line.rstrip() for line in f]
#     except IOError as ercode:
#         if i == 1:
#             print('this is 1')
#     else:
#         print("success")


# def getid(path,id_path):
#     for item in os.listdir(path):
#         print(item[:-4],file = id_path)
# path = '/data1/liumengmeng/SALICON/img_tr'
# id_path = open('/data1/liumengmeng/SALICON/id/train_id.txt','w')
# getid(path,id_path)



# num = random.randint(0,3)
# print(num)


# img = cv2.imread('/data1/liumengmeng/CG4/gt/3b57be29f36bae82a7161dc06486730528abf098_v0.png', -1)
# # print(img.shape)
# # cv2.imwrite('test_.png',img)
# index = np.argwhere(img > 0)
# print(index)

# img = Image.open('/data1/liumengmeng/CG4/img/004ae682bcc180ae5c3da80784808201799f3de6_v2.jpg')
# print(len(img.histogram()))


# print(os.path.exists('/data1/liumengmeng/_data_CG/pic_gt/0aa9efb5cb10ed697ba375d0140b4767855256b6_.png'))
# print(os.path.exists('/data1/liumengmeng/_data_CG/pic_gt/3b57be29f36bae82a7161dc06486730528abf098_.png'))
# print(os.path.exists('/data1/liumengmeng/_data_CG/pic_gt/90de341259f4c9cdd8aa116f50795ab343d0f28b_.png'))
# print(os.path.exists('/data1/liumengmeng/_data_CG/pic_gt/92b8ae6e2efacd9c20dd059a68274364ac938cfa_.png'))


# l1 = txt2list('/data1/liumengmeng/_data_CG/id/contrast_obj_5_obj.txt')
# l2 = txt2list('/data1/liumengmeng/_data_CG/id/contrast_obj_5_bg.txt')

# l1_path = '/data1/liumengmeng/_data_CG/_a_obj_all/'
# for i in l1:
#     shutil.copy(l1_path + i + '.png', './tst/')
# l2_path = '/data1/liumengmeng/_data_CG/_bg_all/'
# for i in l2:
#     shutil.copy(l2_path + i + '.jpg', './tst/' )


l1 = txt2list('/data1/liumengmeng/dataset/MSRA-B/ImageSets/train_id.txt')
l2 = txt2list('/data1/liumengmeng/dataset/MSRA-B/ImageSets/val_id.txt')

l1_path = '/data1/liumengmeng/dataset/MSRA-B/img/'
for i in l1:
    shutil.copy(l1_path + i + '.jpg', '/data1/liumengmeng/dataset/MSRA-B/MSRA-B-train')
# l2_path = '/data1/liumengmeng/_data_CG/_bg_all/'
for i in l2:
    shutil.copy(l1_path + i + '.jpg', '/data1/liumengmeng/dataset/MSRA-B/MSRA-B-val' )
