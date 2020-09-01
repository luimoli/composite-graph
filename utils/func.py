import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil


def getid(path,id_path):
    totallist = []
    for item in os.listdir(path):
        totallist.append(item[:-4])
    list2txt(totallist, id_path)

def list2txt(alist,txt_path):
    with open(txt_path,'w') as f:
        f.write(alist[0])
        for i in range(len(alist)-1):
            f.write('\n'+ alist[i+1])

def txt2list(txt_path):
    with open(txt_path) as f:
        a = [line.rstrip() for line in f]
    return a

def txt2dic(txt_path):
    with open(txt_path,"r",encoding="utf-8") as f:
        idic = json.loads(f.readline())
    return idic

def dic2txt(adic,txt_path):
    with open(txt_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(adic))

def dic2doublelists(dic):
    xname,ynum = [],[]
    for i,v in dic.items():
        xname.append(i)
        ynum.append(v)
    return xname, ynum

def copy_by_idlist(alist,path,new_path, suffix):#按照list保存源目录中的png图片到新目录
    if not os.path.exists(new_path): os.mkdir(new_path)
    for item in alist:
        item_path = os.path.join(path,item + suffix)
        # item_new_path = os.path.join(new_path, item + suffix)
        shutil.copy(item_path,new_path)

def copy_by_idtxt(txtpath, fromfolder, tofolder, suffix):
    if not os.path.exists(tofolder): os.mkdir(tofolder)
    idlist = txt2list(txtpath)
    print('the length of this list:', len(idlist))
    for i in idlist:
        old_path = os.path.join(fromfolder, i + suffix)
        new_path = tofolder
        shutil.copy(old_path,new_path)


def get_obj_sample_list(obj_list,num):
    samplelist = random.sample(obj_list, num)
    return samplelist

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
    if not os.path.exists(cg_root): os.mkdir(cg_root)
    paths = []
    paths.append(cg_root + 'img/')
    paths.append(cg_root + 'gt/')
    paths.append(cg_root + 'instance name/')
    paths.append(cg_root +'instance gt/')
    paths.append(cg_root +'ImageSets/')
    return paths
