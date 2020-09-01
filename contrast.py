import os
import math
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
from utils.func import *
from tqdm import tqdm
# import torch
data_root = '/data1/liumengmeng/_data_CG/'

def get_newobj(fg_path, fg_new_path):#只生成object的像素图
    fg = cv2.imread(fg_path,-1)
    b,g,r,a = cv2.split(fg)
    # index_a_255 = np.argwhere(a == 255)#分出alpha通道，找到其中值为255的点的位置，即可定位在原图中不透明的点的位置
    # num_a = len(np.argwhere(a == 255))#255的像素点的总数量
    index_a_255 = np.argwhere(a > 0)#二值化alpha图，取>0的点
    num_a = len(index_a_255)#255的像素点的总数量
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
# 直方图计算图片相似度
def resize_img(img, size=(256, 256)):
    """所有的图片都统一到256x256"""
    return img.resize(size).convert('RGB')

def hist_similar(lh, rh):
    # assert len(lh) == len(rh)
    # res = []
    # for l,r in zip(lh,rh):
    #     if l == r:
    #         res.append(0)
    #     else:
    #         res.append(float(abs(l-r)) / max(l,r))
    # final = sum(res) / len(lh)
    # return final
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def split_image(img, part_size = (64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() for i in range(0, w, pw) \
            for j in range(0, h, ph)]

# def calc_similar(li, ri):
#     return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0

# def calc_similar_by_path(lf, rf):
#     li, ri = resize_img(Image.open(lf)), resize_img(Image.open(rf))
#     return calc_similar(li, ri)


def calc(lf, ri_his):
    # from ipdb import set_trace;set_trace()
    # li = resize_img(Image.open(lf))
    li = Image.open()
    score = sum(hist_similar(l.histogram(), r) for l, r in zip(split_image(li), ri_his)) / 16.0
    score = round(float(score), 3)
    return score

#---------------------------------------------------------------------------------------
def get_bg_his(bg_path):
    bgid = [] #记录背景图的id
    bghis = [] #记录背景图的切片的直方图lists
    dic = {}
    for item in os.listdir(bg_path):
        # bgid.append(item[:-4])
        img = resize_img(Image.open(os.path.join(bg_path, item)))
        img_split = split_image(img)# split the img into several pieces
        for i in img_split:
            bghis.append(i.histogram())
        dic.update({item[:-4] : bghis})
    print('get background his!')
    return dic


# # 融合函数计算图片相似度
# def calc_image_similarity(img1_path,img2_path):
#     result=float(calc_similar_by_path(img1_path, img2_path))
#     return round(result,3)

def get_lowcontrast_list(obj_path,obj_list,inter_path,dst_path,file1,file2,dic):
    final,maxlist = [],[] #记录和每个object匹配的background的id
    for i in tqdm(obj_list):
        img_path = obj_path + i + '.png'
        img1_path = inter_path + i + '.jpg'
        get_newobj(img_path,img1_path)
        alist,blist = [],[]
        for item in os.listdir(dst_path):
            # img2_path = dst_path + item
            score = calc(img1_path,dic[item[:-4]])
            alist.append(score) #记录对比度得分
            blist.append(item[:-4]) #记录每个得分对应的background的id
        maxvalue = max(alist)
        print(blist[alist.index(maxvalue)],file=file1)
        print(str(maxvalue),file=file2)
        tqdm.write(f'{blist[alist.index(maxvalue)]} = {maxvalue}')

# #object变形的中间图保存路径
inter_path = data_root + 'test_inter/'
src_path = data_root + '_a_obj_all/'
dst_path = data_root + '_bg_all/'
# getid(src_path, '/data1/liumengmeng/_data_CG/id/obj_all.txt')
src_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')
txtname = 'obj_all'
file1 = open(data_root+'id/contrast_'+ txtname + '_dst.txt', 'w')
file2 = open(data_root+'id/contrast_'+ txtname + '_max.txt', 'w')

dic = get_bg_his(dst_path)

get_lowcontrast_list(src_path,src_list,inter_path,dst_path,file1,file2,dic)



# def calc_result(src_list,dst_list,max_list,thresh):
#     num = 0
#     src_n_list,dst_n_list,max_n_list = [],[],[]
#     for i in range(len(max_list)):
#         if max_list[i] >= thresh:
#             num += 1
#             # max_n_list.append(max_list[i])
#             # src_n_list.append(src_list[i])
#             # dst_n_list.append(dst_list[i])
#     return num

#---------------------single test--------------------------------------------------------------------
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
