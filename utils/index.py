import json
from utils.func import *


#----txt and json-----------------------------------------------------------------
def get_txt(name_path):
    with open(name_path) as f:
        a=[line.rstrip() for line in f]
    return a

def get_json(j_path):
    with open(j_path,"r",encoding="utf-8") as f:
        data = json.loads(f.readline())
    return data

def gene_dict_new(img_name,back_name,json_full_path,dict_new_path):
    dic = {}
    dicdata = get_json(json_full_path)#获取所有对应的src图对应的object的type信息
    dic.update({"background":back_name})
    dic_content= {'type' : dicdata[img_name], 'url' : img_name}
    #dic_content= {dicdata[img_name] : img_name}
    dic.update({'1':dic_content})
    with open(dict_new_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))

def gene_dict_new1(img_name,json_full_path,json_old_path,dict_new_path,num):
    dicdata = get_json(json_full_path)
    #old_json = json_old_path + img_name + '.txt'
    with open(json_old_path,"r",encoding="utf-8") as f:
        data = json.loads(f.readline())#读取原合成图（此刻为背景图）的json数据
    dic_content= {'type' : dicdata[img_name], 'url' : img_name}
    #dic_content= {dicdata[img_name] : img_name}
    data.update({num:dic_content})#num表示此次生成是附加上第几个物体
    with open(dict_new_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(data))

def gene_dict_json_form1(img_name,back_name,name_path,dict_path):
    dic = {}
    alist = []
    alist.append({"background":back_name})
    blist = get_txt(name_path)
    for i in range(len(blist)):
        #alist[i+1] = 
        alist.append({i+1:blist[i][2:],"url":img_name})
    dic[img_name] = alist
    with open(dict_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))

def gene_dict(img_name,back_name,name_path,dict_path):
    dic = {}
    alist = {}
    dic.update({"background":back_name})
    blist = get_txt(name_path)
    for i in range(len(blist)):
        #alist[i+1] = 
        alist.update({i+1:blist[i][2:]})
    dic[img_name[:-4]] = alist
    with open(dict_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))

def gene_dict1(img_name,name_path,dict_path,dict_old_path):
    dic = {}#存储最后生成的新json
    alist = {}#存储这次加的object的annotation
    tmp = []#用来计算背景图已有的物体的个数
    dic_old = get_json(dict_old_path)
    dic.update(dic_old)
    #alist.update({"background":back_name})
    blist = get_txt(name_path)#新的font的txt文件的内容
    for i in dic:
        tmp.append(len(dic[i]))
    total = 0
    for i in tmp:
        total += i#json文件里已经有的background和object总数
    if total > 6:
        print(img_name,": over 6 objects")
    for i in range(len(blist)):
        #alist[i+1] = 
        alist.update({i+total:blist[i][2:]})
    dic[img_name[:-4]] = alist
    with open(dict_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))
