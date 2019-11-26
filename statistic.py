import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
data_root = '/data0/liumengmeng/data/'
cg_root = '/data0/liumengmeng/CG/'
def makefolder(paths):#生成路径list对应的文件夹啊
    for path in paths:
        if os.path.exists(path):
            print(path + "  is existed")
        else:
            os.makedirs(path)
            print(path + "  is created!")
def makepath(n):#生成data里的字母文件夹的路径list
    data_root = '/data0/liumengmeng/data/'
    paths = []
    paths.append(data_root + 'pic_res/'+ n +'/')
    paths.append(data_root + 'pic_gt/'+ n + '/')
    paths.append(data_root + 'pic_dic/' + n + '/')
    paths.append(data_root +'pic_ins/' + n + '/')
    return paths
def makepath_final():#生成最终数据集的文件夹
    data_root = '/data0/liumengmeng/CG/'
    paths = []
    paths.append(data_root + 'img/')
    paths.append(data_root + 'gt/')
    paths.append(data_root + 'instance name/')
    paths.append(data_root +'instance gt/')
    return paths

#--------批量删除四个文件夹及其内容-----------------
def removefolder(paths):
    for i in paths:
        shutil.rmtree(i,True)
        print(i,' is removed!')
# removefolder(makepath('c3'))

#---------将一个文件夹下所有文件移动/复制到另外的文件夹-------
def movetoflder(old_folder,new_folder):
    for item in os.listdir(old_folder):
        item_path = os.path.join(old_folder,item)
        shutil.move(item_path,new_folder)
def copytoflder(old_folder,new_folder):
    for item in os.listdir(old_folder):
        item_path = os.path.join(old_folder,item)
        shutil.copy(item_path,new_folder)
def batchcopy(n):
    for i in range(4):
        copytoflder(makepath(n)[i],makepath_final()[i])
# makefolder(makepath_final())
# batchcopy('d3')

def random_pic_id_lists(path,num,nameoftxt):#随机选择合成图并复制到CG文件夹内：
    atotal = []
    for item in os.listdir(path):
        atotal.append(item[:-4])
    blist = random.sample(atotal, num)
    pathoftxt = data_root + 'id/'+ nameoftxt + '.txt'
    for item in blist:
        with open(pathoftxt,'a') as f:
            f.write('\n'+item)
    t = [[],[],[],[]]
    index = ['.jpg','.png','.txt','.png']
    for i in range(4):
        for j in blist:
            t[i].append(j + index[i])
    return t
def copytoflder_list(old_folder,new_folder,alist):
    for item in alist:
        item_path = os.path.join(old_folder,item)
        shutil.copy(item_path,new_folder)
def batchcopy_list(n,num):#n--folder name; num--随机选择的图片的数量
    alists = random_pic_id_lists(makepath(n)[0],num,n)
    for i in range(4):
        copytoflder_list(makepath(n)[i],makepath_final()[i],alists[i])
# batchcopy_list('c5',1000)

#-----------生成所有合成图的id汇总文件并划分训练集与测试集-------------------------
def getid(path,id_path,id_tr_path,id_te_path):
    totallist = []
    tr = []
    te = []
    for item in os.listdir(path):
        totallist.append(item[:-4])
        with open(id_path,'a') as f:
            f.write('\n'+item[:-4])
    te = random.sample(totallist, 6300)
    for i in totallist:
        if not i in te:
            tr.append(i)
    with open(id_tr_path,'a') as f:
        for i in tr:
            f.write('\n'+ i)
    with open(id_te_path,'a') as f:
        for i in te:
            f.write('\n'+ i)

path = cg_root + 'img/'
id_path = cg_root + 'total_id.txt'
id_tr_path = cg_root + 'train_id.txt'
id_te_path = cg_root + 'test_id.txt'
getid(path,id_path,id_tr_path,id_te_path)



#-----------#统计在SOC中一共有多少种类的object--------------------------------
def cal_type(json_path,total_path):
    dicn={}
    dicm={}
    with open(json_path,"r",encoding="utf-8") as f:
        dic = json.loads(f.readline())
    for i in dic:
        dicn.update({dic[i] : 0})
    for i in dic:
        dicn[dic[i]] += 1
    print(len(dic), len(dicn))
    with open(total_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dicn))
# json_path = '/data0/liumengmeng/data/id/instance_json_new.txt'
# total_path = '/data0/liumengmeng/data/id/instance_type.txt'
# cal_type(json_path,total_path)