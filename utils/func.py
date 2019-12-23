import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil

def list2txt(alist,txt_path):
    with open(txt_path,'w') as f:
        f.write(alist[0])
        for i in range(len(alist)-1):
            f.write('\n'+ alist[i+1])

def txt2list(txt_path):
    with open(txt_path) as f:
        a = [line.rstrip() for line in f]
    return a

def savepic_by_list(path,alist,new_path):#按照list保存源目录中的png图片到新目录
    if not os.path.exists(new_path): os.mkdir(new_path)
    for item in alist:
        item_path = os.path.join(path,item + '.png')
        item_new_path = os.path.join(new_path,item + '.png')
        img = cv2.imread(item_path,-1)
        cv2.imwrite(item_new_path,img)

#-----------------------------------------------------------------------------
def makefolder(paths):#生成路径list对应的文件夹啊
    for path in paths:
        if os.path.exists(path):
            print(path + "  is existed")
        else:
            os.makedirs(path)
            print(path + "  is created!")

def makepath(n,data_root):#生成data里的字母文件夹的路径list
    paths = []
    paths.append(data_root + 'pic_res/'+ n +'/')
    paths.append(data_root + 'pic_gt/'+ n + '/')
    paths.append(data_root + 'pic_dic/' + n + '/')
    paths.append(data_root +'pic_ins/' + n + '/')
    return paths

def makepath_final(cg_root):#生成最终数据集的文件夹
    paths = []
    paths.append(cg_root + 'img/')
    paths.append(cg_root + 'gt/')
    paths.append(cg_root + 'instance name/')
    paths.append(cg_root +'instance gt/')
    return paths
