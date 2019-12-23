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
from utils.func import *
from tqdm import tqdm
# import torch
data_root = '/data1/liumengmeng/data_CG/'

def get_newobj(fg_path, fg_new_path):#只生成object的像素图
    fg = cv2.imread(fg_path,-1)
    b,g,r,a = cv2.split(fg)
    index_a_255 = np.argwhere(a == 255)#分出alpha通道，找到其中值为255的点的位置，即可定位在原图中不透明的点的位置
    num_a = len(np.argwhere(a == 255))#255的像素点的总数量
    rl = math.ceil(math.sqrt(num_a))#把不透明的部分的像素点变为一个正方形新图的size
    obj = np.zeros((rl,rl,3))#创建一个三通道的图
    obj = obj.astype(np.uint8)#改变数据类型为uint8
    num = 0
    for i in obj:
        for j in i:
            if num < num_a:
                index = index_a_255[num]
                j[0] = fg[index[0]][index[1]][0]
                j[1] = fg[index[0]][index[1]][1]
                j[2] = fg[index[0]][index[1]][2]
                num += 1
    cv2.imwrite(fg_new_path, obj)
#------------------------------------------------------
# 直方图计算图片相似度算法
def make_regalur_image(img, size=(256, 256)):
    """有必要把所有的图片都统一到特别的规格，在这里256x256的分辨率"""
    return img.resize(size).convert('RGB')

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def split_image(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]

def calc_similar(li, ri):
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0

def calc_similar_by_path(lf, rf):
    li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
    return calc_similar(li, ri)
#---------------------------------------------------------------------------------------
# 融合相似度阈值
threshold1=0.48
# 最终相似度较高判断阈值
threshold2=0.8

# 融合函数计算图片相似度
def calc_image_similarity(img1_path,img2_path):
    """
    :param img1_path: filepath+filename
    :param img2_path: filepath+filename
    :return: 图片最终相似度
    """
    similary_hist=float(calc_similar_by_path(img1_path, img2_path))
    # print('in function:',similary_hist)
    # result = 0
    # if similary_hist > threshold1:
    #     result = similary_hist
    result = similary_hist
    return round(result,3)


def get_obj_sample_list(obj_list,num):
    samplelist = random.sample(obj_list, num)
    return samplelist

def get_lowcontrast_list(obj_path,obj_list,inter_path,dst_path,file1,file2):
    final,maxlist = [],[] #记录和每个object匹配的background的id
    for i in tqdm(obj_list):
        img_path = obj_path + i + '.png'
        img1_path = inter_path + i + '.jpg'
        get_newobj(img_path,img1_path)
        alist,blist = [],[]
        for item in os.listdir(dst_path):
            img2_path = dst_path + item
            kk = calc_image_similarity(img1_path,img2_path)
            alist.append(kk) #记录对比度得分
            blist.append(item[:-4]) #记录每个得分对应的background的id
        maxvalue = max(alist)
        final.append(blist[alist.index(maxvalue)])
        print(blist[alist.index(maxvalue)],file=file1)
        maxlist.append(maxvalue)
        print(maxvalue,file=file2)
        tqdm.write(f'max(alist) = {maxvalue}')
    return final,maxlist

def calc_result(src_list,dst_list,max_list,thresh):
    num = 0
    src_n_list,dst_n_list,max_n_list = [],[],[]
    for i in range(len(max_list)):
        if max_list[i] >= thresh:
            num += 1
            # max_n_list.append(max_list[i])
            # src_n_list.append(src_list[i])
            # dst_n_list.append(dst_list[i])
    return num



# # 搜索图片路径和文件名
# # img1_path='F:/img_spam/data/train/unqrcode/10064003003550210800320010011888.jpg'
# img1_old_path=data_root + '_a_src_full/COCO_train2014_000000001261_2.png'
# img1_path = data_root + 'test_img1/COCO_train2014_000000001261_2.png'
# img_new = get_newobj(img1_old_path)
# cv2.imwrite(img1_path ,img_new)
# # img1_path='./obj_mix.jpg'
# # img1 = make_regalur_image(Image.open(img1_path)).save('test.jpg')
# # 搜索文件夹
# filepath= data_root + '_a_dst/'
# # 相似图片存放路径
# newfilepath = data_root + 'test_simi/'

# alist,pathlist = [],[]
# for parent, dirnames, filenames in os.walk(filepath):
#     for filename in filenames:
#         # print(filepath+filename)
#         img2_path = filepath + filename
#         kk = calc_image_similarity(img1_path,img2_path)
#         alist.append(kk)
#         pathlist.append(img2_path)
# print(max(alist))
# pos = alist.index(max(alist))
# shutil.copy(pathlist[pos],newfilepath)
