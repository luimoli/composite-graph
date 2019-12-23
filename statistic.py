import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
from utils.func import *
data_root = '/data1/liumengmeng/data_CG/'
cg_root = '/data1/liumengmeng/CG2/'
# datas_root = '/data0/liumengmeng/datasets/DUTS/'


#--------批量删除四个文件夹及其内容-----------------
def removefolder(paths):
    for i in paths:
        shutil.rmtree(i,True)
        print(i,' is removed!')
# removefolder(makepath('low_m3',data_root))

#---------将那四个文件夹下所有文件移动/复制到新文件夹-------
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
        copytoflder(makepath(n,data_root)[i],makepath_final(cg_root)[i])
# makefolder(makepath_final(cg_root))
# batchcopy('low_m3')

# copytoflder(datas_root+'imgs/',cg_root+'img/')
# copytoflder(datas_root+'gt/',cg_root+'gt/')
# shutil.copytree(datas_root+'ImageSets',cg_root+'new_id')

#--------------随机选择合成图并复制到CG文件夹内-----------------------------------
def random_pic_id_lists(path,num,nameoftxt):
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

#------------生成所有合成图的id汇总文件并划分训练集与测试集-------------------------
def getid(path,id_path,id_tr_path,id_te_path,train_num):
    totallist,tr,te = [],[],[]
    for item in os.listdir(path):
        totallist.append(item[:-4])
        with open(id_path,'a') as f:
            f.write('\n'+item[:-4])
    tr = random.sample(totallist, train_num)
    for i in totallist:
        if not i in tr:
            te.append(i)
    list2txt(tr,id_tr_path)
    list2txt(te,id_te_path)

# path = cg_root + 'img/'
# id_path = cg_root + 'id/total_id.txt'
# id_tr_path = cg_root + 'id/train_id.txt'
# id_te_path = cg_root + 'id/test_id.txt'
# getid(path,id_path,id_tr_path,id_te_path,16000)

#-----------筛选BO/SO得到id-----------------
def boso(path):
    a,bo,so,sso,pixel = [],[],[],[],[]
    maxnum = 0
    for item in os.listdir(path):
        item_path = os.path.join(path,item)
        img = cv2.imread(item_path)
        b,g,r = cv2.split(img)
        num_a = len(np.argwhere(b == 255))
        if num_a >= 60000 and num_a <= 288747:
            pixel.append(item[:-4])
        if num_a > maxnum:
            maxnum = num_a
        # rate = a.count(255) / len(a)
        # print(item_path,'  ',rate)
        # if rate <= 0.1:
        #     so.append(item[:-4])
        #     if rate <= 0.05:
        #         sso.append(item[:-4])
        # if rate >= 0.5:
        #     bo.append(item[:-4])
        a = []
    print(maxnum)
    return bo,so,sso,pixel
    
# path = data_root + 'tmp/_a_gt_full/'
# list3 = boso(path)
# old_path = data_root+'_a_src_full/'
# new_path = data_root+'pixel/pixel_no_under3000/'
# savepic_by_list(old_path,list3[3],new_path)
# # listtotxt(list3[0],data_root + 'id/bo.txt')
# # listtotxt(list3[1],data_root + 'id/so.txt')
# # listtotxt(list3[2],data_root + 'id/sso.txt')
# list2txt(list3[3],data_root + 'id/pixel_b.txt')

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


#------------对背景图的尺寸进行处理---------------------------------------------
def resize_backgound(path):
    for item in os.listdir(path):
        item_path = os.path.join(path,item)
        img = cv2.imread(item_path)
        r,l = img.shape[:2]
        k =  (600 / min(r,l))
        img_new = cv2.resize(img,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
        # file_ = open('shape_new.txt','w')
        # print(f'old:{img.shape},new:{img_new.shape}', file=file_)
        cv2.imwrite(data_root +'_a_dst_new/'+ item, img_new)
# resize_backgound(data_root+'_a_dst')