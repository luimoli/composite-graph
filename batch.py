import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil
import json
from utils.func import *

def rotate(image):#旋转缩放一定角度
    (h, w) = image.shape[:2]
    # angle = random.randint(-10,10)
    angle = 0
    scale = random.uniform(0.5,1.1)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h)) #h and w 's position
    return rotated

def rotate2images(image,image1):#旋转缩放一定角度
    (h, w) = image.shape[:2]
    angle = random.randint(-10,10)
    scale = random.uniform(0.75,0.95)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h)) #h and w 's position
    rotated1 = cv2.warpAffine(image1, M, (w, h)) #h and w 's position
    return rotated,rotated1

def shape(pic):
    r,c = pic.shape[0:2]
    return r,c

def resize(font,back):#得到按照背景大小缩放后的f_rs
    f_r,f_c = shape(font)
    b_r,b_c = shape(back)
    if b_c/f_c >=1 and b_r/f_r >=1:
        f_rs = font
    else:
        k = min(b_c/f_c, b_r/f_r)
        f_rs = cv2.resize(font,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
    return f_rs

def resize2images(font,back,ins):#得到按照背景大小缩放后的f_rs
    f_r,f_c = shape(font)
    b_r,b_c = shape(back)
    if b_c/f_c >=1 and b_r/f_r >=1:
        f_rs = font
        i_rs = ins
    else:
        k = min(b_c/f_c, b_r/f_r)
        f_rs = cv2.resize(font,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
        i_rs = cv2.resize(ins,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
    return f_rs,i_rs

def get_fnew_and_alpha(f_rs):#缩放过后的前景图的RGB图 #f_new的alpha mask
    b,g,r,a = cv2.split(f_rs)
    f_new = cv2.merge((b,g,r))
    f_new = f_new.astype(float)
    alpha_255 = cv2.merge((a,a,a))
    alpha_ori = alpha_255.copy()
    #二值化alpha图
    index = np.argwhere(alpha_255 > 0)
    for i in index:
        alpha_255[i[0]][i[1]][i[2]] = 255
    return f_new,alpha_255,alpha_ori

#-------确定放在背景图上的位置-------------------------------------------------
def get_bnew(f_rs,back,a,b,c,d,alpha_path):#决定放在背景图的位置
    b_new = back[a:b,c:d].copy()
    b_new = b_new.astype(float)

    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    #alpha_full = cv2.imread(old_a_pth)#读入上一张完整图片的alpha图
    af = alpha_full[a:b,c:d].copy()
    alpha_full[a:b,c:d] = cv2.add(af,get_fnew_and_alpha(f_rs)[1])
    cv2.imwrite(alpha_path, alpha_full)#--------------------------存储gt
    return b_new

def get_bnew1(f_rs,back,a,b,c,d,alpha_path,old_a_pth,ins_path,old_ins_path):#决定放在背景图的位置
    b_new = back[a:b,c:d]
    b_new = b_new.astype(float)

    #alpha_full = np.zeros(back.shape, dtype=back.dtype)
    old_alpha = cv2.imread(old_a_pth)#读入上一张完整图片的alpha图
    # old_ins = cv2.imread(old_ins_path)#读入上一张合成图的ins图
    oa = old_alpha[a:b,c:d].copy()
    # oi = old_ins[a:b,c:d].copy()
    old_alpha[a:b,c:d] = cv2.add(oa,get_fnew_and_alpha(f_rs)[1])
    # old_ins[a:b,c:d] = cv2.add(oi,get_fnew_and_alpha(f_rs)[1])
    cv2.imwrite(alpha_path, old_alpha)
    # cv2.imwrite(ins_path,old_ins)
    return b_new

#--------剪切前景图让其合成到边缘------------------------------------------------
def left(f_rs):#cut f_rs to 40%
    r,c = shape(f_rs)
    f_rs_l = f_rs[:r,int(c*0.4):c]
    return f_rs_l

def right(f_rs):#cut f_rs to 40%
    r,c = shape(f_rs)
    f_rs_r = f_rs[:r,:int(c*0.6)]
    return f_rs_r

def up(f_rs):
    r,c = shape(f_rs)
    f_rs_u = f_rs[int(r*0.4):r,:c]
    return f_rs_u

def down(f_rs):
    r,c = shape(f_rs)
    f_rs_d = f_rs[:int(r*0.6),:c]
    return f_rs_d

#--------------------------------------------------------

def conver(f_rs,b_new):#在背景图上取一块与缩放后的前景图相同的区域与f_rs融合得到f_rs大小的融合图
    f_new,alpha_255,alpha_ori = get_fnew_and_alpha(f_rs)
    alpha = alpha_ori.astype(float)/255#把alpha/255得到的0和1结果用于之后的权数
    f_new = cv2.multiply(alpha,f_new)
    b_new = cv2.multiply(1-alpha,b_new)
    outImage = f_new + b_new
    return outImage

#------------------------------------------------------------------
def choose_pos_center(fp,bp):
    print('this is center!')
    r, c = shape(fp)
    br,bc = shape(bp)
    # size = [[0,r1,0,c1],[(b_r-r1),b_r,0,c1],[0,r1,(b_c-c1),b_c],[(b_r-r1),b_r,(b_c-c1),b_c]]#左上 lu/ 左下 ld / 右上 ru / 右下 rd
    r_ = (br - r)//2
    c_ = (bc - c)//2
    size = [r_, r_+r, c_, c_+c ]
    return size

def choose_pos_random(fp,bp):
    print('this is random!')
    r1, c1 = shape(fp)
    b_r,b_c = shape(bp)
    # size = [[0,r1,0,c1],[(b_r-r1),b_r,0,c1],[0,r1,(b_c-c1),b_c],[(b_r-r1),b_r,(b_c-c1),b_c]] #左上 lu/ 左下 ld / 右上 ru / 右下 rd
    xr = random.randint(0, b_r-r1)
    xc = random.randint(0,b_c-c1)
    size = [xr,xr+r1,xc,xc+c1]
    return size

def pin(font,back,alpha_set,res_set, method):
    f_rs = resize(font,back)
    # a,b,c,d = choose_pos(f_rs,back)
    if method == 'center':
        a,b,c,d = choose_pos_center(f_rs,back)
    elif method == 'random':
        a,b,c,d = choose_pos_random(f_rs,back)
    else:
        print('wrong pin method!')
    b_new = get_bnew(f_rs,back,a,b,c,d,alpha_set)
    outImage = conver(f_rs,b_new)
    res = back.copy()
    res[a:b,c:d] = outImage
    cv2.imwrite(res_set, res)

def pin1(font,back,alpha_path,old_a_pth,ins_path,old_ins_path,res_set):#合成在右下角
    f_rs = resize(font,back)
    a,b,c,d = choose_pos(f_rs,back)
    b_new = get_bnew1(f_rs,back,a,b,c,d,alpha_path,old_a_pth,ins_path,old_ins_path)
    outImage = conver(f_rs,b_new)
    res = back.copy()
    res[a:b,c:d] = outImage
    cv2.imwrite(res_set, res)


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


#-----instance generate----------------------------------------------



#-------------------------------------------------------------------
def batch(src_path, dst_path, res_path, gt_path,name_path,dict_path,name):
    list1 = [];#list里存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    for img_item in os.listdir(src_path):
        #choose font image---------------------------------------
        img_path = os.path.join(src_path, img_item)
        img = cv2.imread(img_path,-1)
        #choose to rotate or not
        src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # src_new = img

        #choose txt-----------------------------------------
        #txt_path = os.path.join(name_path, img_item[:-4]+'.txt')
        json_full_path = os.path.join(name_path, 'instance_json_new.txt')
        #choose ins-----------------------------------------
        # i_path = os.path.join(ins_path, img_item[:-4]+'.png')
        # i_ = cv2.imread(i_path,-1)
        #src_new,i_new = rotate(img,i_)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #choose back----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)#随机读取一张背景图
        
        #set all the write path-----------------------------
        res_set_name = img_item[:-4]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')#设定存储gt的完整路径和命名
        dic_set = os.path.join(dict_path, res_set_name +'.txt')#设定存储dicts的完整路径和命名
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #ins_set = os.path.join(color_path, res_set_name +'.png')
        #funcs-----------------------------------------------
        pin(src_new,dst_new,al_set,res_set,flag)#存储合成图并存储了gt
        #gene_dict(img_item, list1_[l_num] , txt_path , dic_set)
        gene_dict_new(img_item[:-4] ,list1_[l_num],json_full_path,dic_set)

def batch1(src_path, dst_path, res_path, gt_path,gt_old_path,ins_path,ins_old_path,name_path,dict_path,dict_old_path,name,num):
    list1 = [];
    list2 = [];
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)#存背景图的完整路径
        list2.append(dst_item)#存背景图的名称
    total = len(list1)
    for img_item in os.listdir(src_path):
        img_path = os.path.join(src_path, img_item)
        img = cv2.imread(img_path,-1)
        src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #src_new = img

        json_full_path = os.path.join(name_path, 'instance_json_new.txt')

        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)#选取一张随机的背景图(合成图)

        res_set_name = img_item[:-4]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')#设定存储gt的完整路径和命名
        dic_set = os.path.join(dict_path, res_set_name +'.txt')#设定存储dicts的完整路径和命名
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        old_dic_path = os.path.join(dict_old_path,list2[l_num][:-4]+'.txt')#找到此背景图的json信息
        old_a_path = os.path.join(gt_old_path, list2[l_num][:-4]+'.png')#找到此背景图的gt(已经是合成图)
        ins_set = os.path.join(ins_path, res_set_name +'.png')
        old_ins_path = os.path.join(ins_old_path, list2[l_num][:-4]+'.png')
        #pin1(src_new, dst_new, al_set, old_a_path, res_set)#生成新合成图并存储了新的gt
        pin1(src_new,dst_new,al_set,old_a_path,ins_set,old_ins_path,res_set,flag)
        gene_dict_new1(img_item[:-4],json_full_path,old_dic_path,dic_set,num)

def batch_contrast(src_path,src_list, dst_path, dst_list,res_path, gt_path,name):
    for i in range(len(src_list)):
        #choose font image---------------------------------------
        img_path = os.path.join(src_path, src_list[i]+'.png')
        img = cv2.imread(img_path,-1)
        #choose to rotate or not
        src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #src_new = img

        #choose back----------------------------------------
        back_path = os.path.join(dst_path, dst_list[i]+'.jpg')
        dst_new = cv2.imread(back_path,-1)
        
        #set all the write path-----------------------------
        res_set_name = src_list[i]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')#设定存储gt的完整路径和命名
        # dic_set = os.path.join(dict_path, res_set_name +'.txt')#设定存储dicts的完整路径和命名
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #ins_set = os.path.join(color_path, res_set_name +'.png')
        #funcs-----------------------------------------------
        pin(src_new,dst_new,al_set,res_set,'random')
        # gene_dict_new(src_list[i] ,dst_list[i], json_full_path,dic_set)

def batch_no_random_bg(src_path, dst_path, res_path, gt_path,name,center_num):
    list1 = [];#存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    pos = 0
    tmp = 0
    for img_item in os.listdir(src_path):
        #choose font image---------------------------------------
        img_path = os.path.join(src_path, img_item)
        img = cv2.imread(img_path,-1)
        #choose to rotate or not
        src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # src_new = img
        #choose back----------------------------------------
        # l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[pos],-1)#随机读取一张背景图
        pos +=1
        #set all the write path-----------------------------
        res_set_name = img_item[:-4]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')#设定存储gt的完整路径和命名
        # dic_set = os.path.join(dict_path, res_set_name +'.txt')#设定存储dicts的完整路径和命名
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #ins_set = os.path.join(color_path, res_set_name +'.png')
        #funcs-----------------------------------------------
        if tmp < center_num:
            pin(src_new,dst_new,al_set,res_set,'center')#存储合成图并存储了gt
        else:
            pin(src_new,dst_new,al_set,res_set,'random')
        tmp += 1
        # gene_dict_new(img_item[:-4] ,list1_[l_num],json_full_path,dic_set)
#-----------------------------------------------------------------------------------------------

#---------------------------------------
data_root = '/data1/liumengmeng/_data_CG/'
#--------------------------------------

#------------------------------------------------------------------------
#-------生成单个objec合成图--------------
# n = 'b3'#新的合成图要存放的文件夹
# name = n #新的合成图的命名
# flag = 3
# src_path = data_root + '_a_src_te/'
# dst_path = data_root + '_a_dst_te/'
# name_path = data_root + 'id/'
# paths = makepath(n, data_root)
# makefolder(paths)
# # batch(src_path, dst_path, res_path, gt_path,name_path,dict_path,name,flag)
# batch(src_path, dst_path, paths[0], paths[1],name_path,paths[2],name,flag)
#-----------------------------------------------------------------------

# # --------生成多个objec合成图------------------
# n = 'b3_0'#新的合成图要存放的文件夹
# name = n #新的合成图的命名
# m = 'b3'#作为背景图的原来的合成图的文件夹
# num = '2'#在json标注中是第几个物体
# flag = 0
# src_path = data_root + '_a_src_te/'
# name_path = data_root + 'id/'
# pm = makepath(m,data_root)
# pn = makepath(n,data_root)
# makefolder(pn)
# batch1(src_path, pm[0], pn[0], pn[1],pm[1],pn[3],pm[3],name_path,pn[2],pm[2],name,num,flag)


#------------------------------------------------------------------------
#-------生成背景不随机的单个objec合成图--------------
# n = 'v2'#新的合成图要存放的文件夹
# name = n #新的合成图的命名
# src_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
# center_num = 4480
# paths = makepath(n, data_root)
# makefolder(paths)
# batch_no_random_bg(src_path, dst_path, paths[0], paths[1], name, center_num)
#-----------------------------------------------------------------------


#------------低对比度图------------------------------------------
n = 'low_501'#新的合成图要存放的文件夹
name = n #新的合成图的命名
src_path = data_root + '_a_obj_all/'
dst_path = data_root + '_bg_all/'
paths_contra = makepath(n,data_root)
makefolder(paths_contra)
#-----根据id合成低对比度图-------------------
src_list = txt2list(data_root+'id/contrast_obj_501_obj.txt')
dst_list = txt2list(data_root+'id/contrast_obj_501_bg.txt')
batch_contrast(src_path, src_list, dst_path, dst_list, paths_contra[0], paths_contra[1],name)










#test-----------------------------------------------------------------
# fg = cv2.imread(data_root+"_a_src_full/COCO_train2014_000000000332.png",-1)
# bg = cv2.imread(data_root+"_a_dst/bing_bg_1_0105.jpg")
# ins = cv2.imread(data_root+"_a_ins/COCO_train2014_000000000853.png",-1)
# gt_4667 = cv2.imread(data_root+"_a_gt_full/COCO_train2014_000000000110_3.png",-1)
