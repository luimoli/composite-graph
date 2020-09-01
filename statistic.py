import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
from utils.func import *
import matplotlib.pyplot as plt
data_root = '/data1/liumengmeng/_data_CG/'
cg_root = '/data1/liumengmeng/CG4DUTSTR/'


#--------批量删除四个文件夹及其内容-----------------
def removefolder(paths):
    for i in paths:
        shutil.rmtree(i,True)
        print(i,' is removed!')
# removefolder(makepath('h4_test',data_root))

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
# batchcopy('m4')

#从pic_res的结果里挑出来一些合适的图，通过id复制到CG_ROOT的文件夹里
def get_selectfolder_list(select_folder):
    totallist = []
    for item in os.listdir(select_folder):
        totallist.append(item[:-4])
    return totallist

def copytoflder_by_list(select_folder, old_folder, new_folder,suffix):
    slist = get_selectfolder_list(select_folder)
    for i in slist:
        ipath = os.path.join(old_folder,i+suffix)
        shutil.copy(ipath,new_folder)
def batch_selectcopy(select_foldername,n):
    # for i in range(2):
    copytoflder_by_list(makepath(select_foldername,data_root)[0],makepath(n,data_root)[0],makepath_final(cg_root)[0],'.jpg')
    copytoflder_by_list(makepath(select_foldername,data_root)[0],makepath(n,data_root)[1],makepath_final(cg_root)[1],'.png')

# makefolder(makepath_final(cg_root))
# batch_selectcopy('h1_batch2s','h1_batch2')



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

#------------生成所有合成图的id汇总文件并随机划分训练集与测试集-------------------------
def getid_tr_te(path,id_path,id_tr_path,id_te_path,train_num):
    totallist,tr,te = [],[],[]
    for item in os.listdir(path):
        totallist.append(item[:-4])
    list2txt(totallist, id_path)
    # tr = random.sample(totallist, train_num)
    # for i in totallist:
    #     if not i in tr:
    #         te.append(i)
    # list2txt(tr,id_tr_path)
    # list2txt(te,id_te_path)

def getid_(path,id_path):
    totallist = []
    for item in os.listdir(path):
        totallist.append(item[:-4])
    list2txt(totallist, id_path)

path = cg_root + 'img/'
id_path = cg_root + 'ImageSets/total_id.txt'
# path = '/data1/liumengmeng/CG4_test/DUTS-TR/img'
# id_path = '/data1/liumengmeng/CG4_test/DUTS-TR/ImageSets/total_id.txt'

# getid_(path,id_path)


#-----------筛选BO/SO得到id----【依据像素点】-------------
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
# copy_by_idlist(list3[3],old_path,new_path)
# # listtotxt(list3[0],data_root + 'id/bo.txt')
# # listtotxt(list3[1],data_root + 'id/so.txt')
# # listtotxt(list3[2],data_root + 'id/sso.txt')
# list2txt(list3[3],data_root + 'id/pixel_b.txt')

# =================================================================================================================

def plot_his(x,y):
    plt.figure(figsize=(5,15))#设置画布的尺寸
    plt.title('the objects of CG2',fontsize=20)#标题，并设定字号大小
    plt.xlabel(u'num',fontsize=14)#设置x轴，并设定字号大小
    plt.ylabel(u'name',fontsize=14)#设置y轴，并设定字号大小
    #alpha：透明度；facecolor：柱子填充色；edgecolor：柱子轮廓色；lw：柱子轮廓的宽度；label：图例；
    plt.barh(x,y, alpha=0.6, facecolor = 'deeppink', edgecolor = 'deeppink', label='Jay income')
    # plt.legend(loc=4)#图例展示位置，数字代表第几象限
    plt.savefig(os.path.join('test2.png'), dpi=300, format='png', bbox_inches='tight')

def plot_dic_his(dic):
    xname,ynum = [],[]
    for i,v in dic.items():
        xname.append(i)
        ynum.append(v)
    plot_his(xname,ynum)

#-----------#统计在SOC中一共有多少种类的object--------------------------------
def cal_type(json_path, total_path):
    dicn={}
    dic = txt2dic(json_path)
    for i in dic:
        dicn.update({dic[i] : 0})
    for i in dic:
        dicn[dic[i]] += 1
    print(len(dic), len(dicn))
    dic2txt(dicn, total_path)

# json_path = '/data1/liumengmeng/_data_CG/id/instance_json_3971.txt'
# total_path = '/data1/liumengmeng/_data_CG/id/instance_type_3971.txt'
# cal_type(json_path, total_path)


# dic = txt2dic(total_path)
# print(len(dic))
# print(dic['person'])
# plot_dic_his(dic)


#----------统计在CG2中不同类别的物体的数量--------------------------------------------------
def sumall(insname_path):
    dicn={}
    xname,ynum = [],[]
    for item in os.listdir(insname_path):
        item_path = os.path.join(insname_path,item)
        with open(item_path,"r",encoding="utf-8") as f:
            idic = json.loads(f.readline())
        for i,t in idic.items():
            if i == 'background':
                pass
            else:
                dicn.update({t['type'] : 0})
    # print(dicn)
    for item in os.listdir(insname_path):
        item_path = os.path.join(insname_path,item)
        with open(item_path,"r",encoding="utf-8") as f:
            idic = json.loads(f.readline())
        for i,t in idic.items():
            if i == 'background':
                pass
            else:
                dicn[t['type']] += 1
    # print(dicn)
    # return dicn
    for i,v in dicn.items():
        xname.append(i)
        ynum.append(v)
    return xname, ynum

# insname_path = '/data1/liumengmeng/dataset/CG2/instance name'     
# x,y = sumall(insname_path)
# plot_his(x,y)




#------------对背景图的尺寸进行处理---------------------------------------------
def resize_backgound(path):
    for item in os.listdir(path):
        item_path = os.path.join(path,item)
        img = cv2.imread(item_path)
        r,l = img.shape[:2]
        k =  (560 / min(r,l))
        img_new = cv2.resize(img,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
        # file_ = open('shape_new.txt','w')
        # print(f'old:{img.shape},new:{img_new.shape}', file=file_)
        cv2.imwrite(data_root +'_a_dst_new/'+ item, img_new)

# resize_backgound(data_root+'_a_dst')

#---------------------将背景图分为一半/一半-----------------------------
def break_background(path,new_path_tr,new_path_te):
    totallist,back_tr,back_te = [],[],[]
    for item in os.listdir(path):
        totallist.append(item[:-4])
    # list2txt(totallist, id_path)
    # print(len(totallist))
    num = int(len(totallist)/2)
    back_tr = random.sample(totallist, num)
    for i in totallist:
        if not i in back_tr:
            back_te.append(i)
    # list2txt(tr,id_tr_path)
    copy_by_idlist(back_tr, path, new_path_tr, '.jpg')
    copy_by_idlist(back_te, path, new_path_te, '.jpg')

# path = '/data1/liumengmeng/_data_CG/_a_dst'
# new_path_tr = '/data1/liumengmeng/_data_CG/_a_dst_tr'
# new_path_te = '/data1/liumengmeng/_data_CG/_a_dst_te'
# break_background(path,new_path_tr,new_path_te)

#------------统计某个type的object的所有图片的id-------------------------------------------
def find_object(id_path, object_name, id1, id2):
    for item in os.listdir(id_path):
        item_path = os.path.join(id_path, item)
        with open(item_path) as f:
            a = [line.rstrip() for line in f]
        num = 0
        for i in a:
            if object_name in i:
                num += 1
        if num == len(a):
            print(item[:-4], file=id1)
        elif num == 0:
            print(item[:-4], file=id2)

# id_path = '/data1/liumengmeng/data_CG/tmp/_a_name'     
# object_name = 'person'
# id1 = open('/data1/liumengmeng/data_CG/tmp/person.txt','w')
# id2 = open('/data1/liumengmeng/data_CG/tmp/person_none.txt', 'w')
# find_object(id_path, object_name, id1, id2)

#----------get total 3971 objects' id_file---------------------------------
# a = txt2list('/data1/liumengmeng/_data_CG/id/pixel_b.txt')
# b = txt2list('/data1/liumengmeng/_data_CG/id/pixel_m.txt')
# c = txt2list('/data1/liumengmeng/_data_CG/id/pixel_s.txt')
# total = a + b + c
# # print(len(total))
# list2txt(total, '/data1/liumengmeng/_data_CG/id/total_3971.txt')

#---------获取并统计3971 objects中的物体类型-------------------------------------------
# instance_json_4667 = '/data1/liumengmeng/_data_CG/id/instance_json_new.txt'
# dic = txt2dic(instance_json_4667)
# alist = txt2list('/data1/liumengmeng/_data_CG/id/total_3971.txt')
# dic_3971 = {}
# for i,v in dic.items():
#     if i in alist:
#         dic_3971.update({i : v})
# print(dic_3971)
# dic2txt(dic_3971, '/data1/liumengmeng/_data_CG/id/instance_json_3971.txt')

#-------将object type总数文件排序，对半分成两份，返回type names的两个数组，分别size是41，41-------------------
def get41_41typename(type_txt_path):
    dic = txt2dic(type_txt_path)
    dic_sorted_84 = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]))
    # print(dic_sorted[1])
    dic_sorted_82 = []
    for i in range(len(dic_sorted_84)-2):
        dic_sorted_82.append(dic_sorted_84[i+1])
    print(len(dic_sorted_82))

    tr,te = [],[] #存储字典排序后的tuple值
    for i in range(int(len(dic_sorted_82) / 2 )):
        te.append(dic_sorted_82[2*i]) # 测试集 --偶数个
        tr.append(dic_sorted_82[2*i + 1]) # 训练集 --奇数个
    te_, tr_ = [],[] #存储对半分之后的type name
    for i in te:
        te_.append(i[0])
    for i in tr:
        tr_.append(i[0])
    return tr_,te_

# type_txt_path = '/data1/liumengmeng/_data_CG/id/instance_type_3971.txt'
# tr, te = get41_41typename(type_txt_path)
# # print(tr)
# # print(te)
# tr.remove('sandwich')
# tr.append('bowl')
# te.remove('bowl')
# te.append('sandwich')

# #-----根据分类的type name的两个list 将src object分开，得到id---------------------------
# ins_txt_path = '/data1/liumengmeng/_data_CG/id/instance_json_3971.txt'
# adic = txt2dic(ins_txt_path)
# tr_id, te_id = [],[]
# num = 0
# for i,v in adic.items():
#     if v in tr:
#         tr_id.append(i)
#     elif v in te:
#         te_id.append(i)
#     else:
#         num += 1
# print(len(tr_id),len(te_id),num)
# list2txt(tr_id, '/data1/liumengmeng/_data_CG/id/tr_id2_1200.txt')
# list2txt(te_id, '/data1/liumengmeng/_data_CG/id/te_id2_1139.txt')

# trpath = '/data1/liumengmeng/_data_CG/id/tr_id2_1200.txt'
# tepath = '/data1/liumengmeng/_data_CG/id/te_id2_1139.txt'
# copy_by_idtxt(trpath, '/data1/liumengmeng/_data_CG/_a_src_full', '/data1/liumengmeng/_data_CG/_a_src_tr', '.png')
# copy_by_idtxt(tepath, '/data1/liumengmeng/_data_CG/_a_src_full', '/data1/liumengmeng/_data_CG/_a_src_te', '.png')





# #------------验证是不是分错了-----------------------------------
# path = '/data1/liumengmeng/CG3-TR/instance name'
# for item in os.listdir(path):
#     item_path = os.path.join(path,item)
#     idic = txt2dic(item_path)
#     for i,t in idic.items():
#         if i == 'background':
#             pass
#         else:
#             if t['type'] == 'bus':
#                 print('found it')
#             # else:
#             #     print(t['type'],'no')


