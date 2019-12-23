import os
import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
data_root = '/data0/liumengmeng/data/'

# path = os.path.expanduser('~')
# print(path)

# fg = cv2.imread(data_root + "_a_src/COCO_train2014_000000001510.png",-1)
# ins = cv2.imread(data_root + "_a_ins/COCO_train2014_000000052270.png",-1)
# cv2.imwrite("ins_original.png",ins)
# cv2.imwrite("fg_test.png",fg)

def get_dic(name_path):#把SOC中的instance name拆成字典并返回一个字典
    adic = {}
    with open(name_path) as f:
        a=[line.rstrip() for line in f]
    for i in range(len(a)):
        adic.update({i+1 : a[i][2:]})
    return adic

def srcjpg_multi(src_path,name_path,new_path,dict_path):#remove different pics between the two folders
    dic = {}
    for ins_item in os.listdir(src_path):
        ins_path = os.path.join(src_path, ins_item)
        txt_path = os.path.join(name_path,ins_item[:-4]+'.txt')
        ins = cv2.imread(ins_path)
        ins_dic = get_dic(txt_path)#得到此时的src对应的instance name 的字典
        dic.update({ins_item[:-4] : ins_dic})#保存新字典，新的字典的格式{图片名：{字典}}
        #生成对应4667张的原来的coco的jpg图
        if(len(ins_dic) == 1):
            shutil.copy(ins_path , new_path+'/'+ ins_item)
        else:
            for i in range(len(ins_dic)):
                shutil.copy(ins_path , new_path+'/'+ ins_item[:-4] +'_'+ str(i+1) +'.jpg')
    with open(dict_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))

def get_new_dic(old_json_path,new_json_path):#生成新的len为4667的字典
    dicn={}
    with open(old_json_path,"r",encoding="utf-8") as f:
        dic = json.loads(f.readline())
    for i in dic:
        if(len(dic[i]) == 1):
            #print('==1:',dic[i])
            dicn.update({i:dic[i]['1']})
        else:
            for j in range(len(dic[i])):
                #print('==several', dic[i][str(j+1)])
                dicn.update({i+'_'+str(j+1) : dic[i][str(j+1)] })
    print(len(dic), len(dicn))
    with open(new_json_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dicn))
#---------------------------------------------------------------------------------------

src_path = '/data0/liumengmeng/data/_a_src_jpg'
name_path = '/data0/liumengmeng/data/_a_name'
new_path = '/data0/liumengmeng/data/_a_src_full'
dict_path = '/data0/liumengmeng/data/id/instance_json.txt'
# srcjpg_multi(src_path,name_path,new_path,dict_path)

old_json_path = '/data0/liumengmeng/data/id/instance_json.txt'
new_json_path = '/data0/liumengmeng/data/id/instance_json_new.txt'
get_new_dic(old_json_path,new_json_path)


#----------------test----------------------
# b,g,r = cv2.split(ins)
# z = np.zeros(b.shape, dtype=b.dtype)
# y = cv2.add(g,r)
# b_new_1 = cv2.merge((b,z,z))
# b_new_2 = cv2.merge((z,g,z))
# b_new_255 = cv2.merge((z,g,g))
# cv2.imwrite("blue_1.png",b_new_1)
# cv2.imwrite("blue_2.png",b_new_2)
# cv2.imwrite("ins_255.png",b_new_255)
