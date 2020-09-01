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
    scale = random.uniform(0.5,1.0)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h)) #h and w 's position
    return rotated

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

def resize_part(font,back,bg_part):#得到按照背景大小缩放后的f_rs
    f_r,f_c = shape(font)
    b_r,b_c = shape(back)
    if b_r < b_c:
        k = b_c / bg_part / f_c
        k1 = b_r / f_r
        k = min(k,k1)
        flag = 'p'
    else:
        k = b_r / bg_part / f_r
        k1 = b_c / f_c
        k = min(k,k1)
        flag = 'v'
    f_rs = cv2.resize(font,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
    return f_rs,flag

def resize_bs_part(fontlist,back,fraction):
    f_r,f_c = shape(fontlist[0])
    f_r1,f_c1 = shape(fontlist[1])
    b_r,b_c = shape(back)
    if b_r < b_c:
        kb = b_c * fraction / f_c
        kb1 = b_r / f_r
        kb = min(kb,kb1)
        ks = b_c * (1-fraction) / f_c1
        ks1 = b_r / f_r1
        ks = min(ks,ks1)
        f_rs_b = cv2.resize(fontlist[0],None,fx=kb,fy=kb, interpolation = cv2.INTER_CUBIC)
        f_rs_s = cv2.resize(fontlist[1],None,fx=ks,fy=ks, interpolation = cv2.INTER_CUBIC)
        flag = 'p'
    else:
        kb = b_r * fraction / f_r
        kb1 = b_c / f_c
        kb = min(kb,kb1)
        ks = b_r * (1-fraction) / f_r1
        ks1 = b_c / f_c1
        ks = min(ks,ks1)
        f_rs_b = cv2.resize(fontlist[0],None,fx=kb,fy=kb, interpolation = cv2.INTER_CUBIC)
        f_rs_s = cv2.resize(fontlist[1],None,fx=ks,fy=ks, interpolation = cv2.INTER_CUBIC)
        flag = 'v'
    return f_rs_b,f_rs_s,flag


def get_fraction_k(f_r,f_c,b_r,b_c,fraction_part):# set the value of fraction to get the resized object
    if b_r < b_c:
        k = b_c / fraction_part / f_c
        k1 = b_r / f_r
        k = min(k,k1)
    else:
        k = b_r / fraction_part / f_r
        k1 = b_c / f_c
        k = min(k,k1)
    return k

def resize_2_distance(fontlist,back,fraction_part):
    f_r,f_c = shape(fontlist[0])
    f_r1,f_c1 = shape(fontlist[1])
    b_r,b_c = shape(back)
    k = get_fraction_k(f_r,f_c,b_r,b_c,fraction_part)
    k1 = get_fraction_k(f_r1,f_c1,b_r,b_c,fraction_part)
    f_rs = cv2.resize(fontlist[0],None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
    f_rs1 = cv2.resize(fontlist[1],None,fx=k1,fy=k1, interpolation = cv2.INTER_CUBIC)
    if b_r < b_c:
        flag = 'p'
    else:
        flag = 'v'
    return f_rs,f_rs1,flag


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
def get_bnew(f_rs,back,a,b,c,d,alpha_full):#决定放在背景图的位置
    b_new = back[a:b,c:d].copy()
    b_new = b_new.astype(float)

    af = alpha_full[a:b,c:d].copy()
    alpha_full[a:b,c:d] = cv2.add(af,get_fnew_and_alpha(f_rs)[1])
    # cv2.imwrite(alpha_path, alpha_full)#---------存储gt
    return b_new, alpha_full #根据font的bg切片  /  新的gt


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
    r_ = (br - r) // 2
    c_ = (bc - c) // 2
    size = [r_, r_+r, c_, c_+c ]
    return size

def choose_pos_random(fp,bp):
    print('this is random!')
    r1, c1 = shape(fp)
    b_r,b_c = shape(bp)
    xr = random.randint(0, b_r-r1)
    xc = random.randint(0,b_c-c1)
    size = [xr,xr+r1,xc,xc+c1]
    return size

#----------------------new pos---------------------
def choose_pos_random_part(fp,bp,bg_part,flag):
    print('this is random for part!')
    r1, c1 = shape(fp)
    b_r,b_c = shape(bp)
    size = []
    if flag == 'p':
        xr = random.randint(0, b_r-r1)
        tmp = int(b_c / bg_part)
        for i in range(bg_part):
            # xc = random.randint(tmp*i ,tmp*(i+1))
            xc = tmp*i
            size.append([xr,xr+r1,xc,xc+c1])
    elif flag == 'v':
        xc = random.randint(0,b_c-c1)
        tmp = int(b_r / bg_part)
        for i in range(bg_part):
            # xr = random.randint(tmp*i ,tmp*(i+1))
            xr = tmp*i
            size.append([xr,xr+r1,xc,xc+c1])
    else:
        print('wrong flag about _choose_pos_random_part !')
    return size

def choose_pos_big_small(fpb,fps, bp, fraction, flag):
    print('this is for big plus small objects!')
    r1, c1 = shape(fpb)
    r2, c2 = shape(fps)
    b_r,b_c = shape(bp)
    size = []
    if flag == 'p':
        rannum = random.randint(0,1) # flag表示两个物体如何放置
        if rannum == 0:
            tmp = int(b_c * fraction)
            xc = 0
            xr = random.randint(0, b_r-r1)
            size.append([xr,xr+r1,xc,xc+c1])
            xc = tmp
            xr = random.randint(0, b_r-r2)
            size.append([xr,xr+r2,xc,xc+c2])
        elif rannum == 1:
            tmp1 = int(b_c * (1-fraction))
            xc = tmp1
            xr = random.randint(0, b_r-r1)
            size.append([xr,xr+r1,xc,xc+c1])
            xc = 0
            xr = random.randint(0, b_r-r2)
            size.append([xr,xr+r2,xc,xc+c2])    

    elif flag == 'v':
        rannum = random.randint(0,1) # flag表示两个物体如何放置
        if rannum == 0:
            tmp = int(b_r * fraction)
            xr = 0
            xc = random.randint(0, b_c-c1)
            size.append([xr,xr+r1,xc,xc+c1])
            xr = tmp
            xc = random.randint(0, b_c-c2)
            size.append([xr,xr+r2,xc,xc+c2])
        elif rannum == 1:
            tmp1 = int(b_r * (1-fraction))
            xr = tmp1
            xc = random.randint(0, b_c-c1)
            size.append([xr,xr+r1,xc,xc+c1])
            xr = 0
            xc = random.randint(0, b_c-c2)
            size.append([xr,xr+r2,xc,xc+c2]) 
    else:
        print('wrong flag about _choose_pos_random_part !')
    return size

def choose_pos_distance_random(fp,fp1, bp, fraction_part, flag):
    print('this is for distance!')
    r, c = shape(fp)
    r1, c1 = shape(fp1)
    b_r,b_c = shape(bp)
    size = []
    if flag == 'p':
        tmp = int(b_c / fraction_part * (fraction_part - 1))
        xc = 0
        xr = random.randint(0, b_r-r)
        size.append([xr,xr+r,xc,xc+c])
        xc = tmp
        xr = random.randint(0, b_r-r1)
        size.append([xr,xr+r1,xc,xc+c1])
    elif flag == 'v':
        tmp = int(b_r / fraction_part * (fraction_part - 1))
        xr = 0
        xc = random.randint(0, b_c-c)
        size.append([xr,xr+r,xc,xc+c])
        xr = tmp
        xc = random.randint(0, b_c-c1)
        size.append([xr,xr+r1,xc,xc+c1])
    return size

def choose_pos_distance_center(fp,fp1, bp, fraction_part, flag):
    print('this is for distance!')
    r, c = shape(fp)
    r1, c1 = shape(fp1)
    b_r,b_c = shape(bp)
    size = []
    if flag == 'p':
        tmp = int(b_c / fraction_part * (fraction_part - 1))
        xc = 0
        xr = (b_r - r) // 2
        size.append([xr,xr+r,xc,xc+c])
        xc = tmp
        xr = (b_r - r1) // 2
        size.append([xr,xr+r1,xc,xc+c1])
    elif flag == 'v':
        tmp = int(b_r / fraction_part * (fraction_part - 1))
        xr = 0
        xc = (b_c - c) // 2
        size.append([xr,xr+r,xc,xc+c])
        xr = tmp
        xc = (b_c - c1) // 2
        size.append([xr,xr+r1,xc,xc+c1])
    return size

#------------------------------------------------------------------------------------------------------------------
def pin(font,back,alpha_set,res_set,obj_num,bg_part):
    # f_rs = resize(font,back)
    f_rs,flag = resize_part(font,back,bg_part)

    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    for i in range(obj_num):#------重复拼贴object-------------------
        # a,b,c,d = choose_pos_random(f_rs,back)
        a,b,c,d =  choose_pos_random_part(f_rs,back,bg_part,flag)[i]
        b_new,alpha_new = get_bnew(f_rs,back,a,b,c,d,alpha_full)
        outImage = conver(f_rs,b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        alpha_full = alpha_new
        back = res
    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)

def pin_smalls(fontlist,back,alpha_set,res_set,bg_part):
    # f_rs = resize(font,back)
    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    for i in range(len(fontlist)):
        f_rs,flag = resize_part(fontlist[i], back, bg_part)
        a,b,c,d =  choose_pos_random_part(f_rs,back,bg_part,flag)[i]
        b_new,alpha_new = get_bnew(f_rs,back,a,b,c,d,alpha_full)
        outImage = conver(f_rs,b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        alpha_full = alpha_new.copy()
        back = res.copy()
    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)

def pin_big_small(fontlist,back,alpha_set,res_set,fraction):
    # f_rs = resize(font,back)
    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    f_rs_b,f_rs_s,flag = resize_bs_part(fontlist,back,fraction)
    size = choose_pos_big_small(f_rs_b,f_rs_s,back,fraction, flag)
    fontlist = [f_rs_b,f_rs_s]
    for i in range(len(fontlist)):
        a, b, c, d = size[i]
        b_new,alpha_new = get_bnew(fontlist[i],back,a,b,c,d,alpha_full)
        outImage = conver(fontlist[i],b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        alpha_full = alpha_new.copy()
        back = res.copy()
    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)

def pin_distance(fontlist,back,alpha_set,res_set,fraction_part):
    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    f_rs_b,f_rs_s,flag = resize_2_distance(fontlist,back,fraction_part)
    size = choose_pos_distance_random(f_rs_b,f_rs_s,back,fraction_part, flag)#--------TODO:method can be changed
    # print(size)
    fontlist = [f_rs_b,f_rs_s]
    for i in range(len(fontlist)):
        a, b, c, d = size[i]
        b_new,alpha_new = get_bnew(fontlist[i],back,a,b,c,d,alpha_full)
        outImage = conver(fontlist[i],b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        alpha_full = alpha_new.copy()
        back = res.copy()
    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)


def get_obj_sample_list(obj_list,num):
    samplelist = random.sample(obj_list, num)
    return samplelist

#-------------------------------------------------------------------

def batch_same_obj(src_path, obj_list, dst_path, res_path, gt_path,name, obj_num,bg_part):
    list1 = [];#list里存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    for img_item in obj_list:
        #choose font image---------------------------------------
        img_path = os.path.join(src_path, img_item + '.png')
        img = cv2.imread(img_path,-1)
        #choose to scale or not----------------------------------
        # src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        src_new = img
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1) 
        #set all the write path-----------------------------
        res_set_name = img_item[:-4]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs-----------------------------------------------
        pin(src_new,dst_new,al_set,res_set, obj_num, bg_part)

def batch_smalls(src_path, obj_list, dst_path, res_path, gt_path,name,bg_part):
    list1 = [];#list里存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    for i in range(0,len(obj_list),bg_part):
        #choose font image---------------------------------------
        # fontlist = [obj_list[i] for j in range(i, i+bg_part)]
        fontlist = [cv2.imread(os.path.join(src_path, obj_list[j] + '.png'),-1) for j in range(i, i+bg_part)]
        # img_path = os.path.join(src_path, img_item + '.png')
        # img = cv2.imread(img_path,-1)
        #choose to scale or not----------------------------------
        # src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # src_new = img
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1) 
        #set all the write path-----------------------------
        # res_set_name = img_item[:-4]+'_'+ name
        res_set_name = list1_[l_num]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs-----------------------------------------------
        pin_smalls(fontlist,dst_new,al_set,res_set,bg_part)

def batch_2_big_small(src_path, obj_list, dst_path, res_path, gt_path,name,read_obj_num,fraction):
    list1 = [];#list里存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    for i in range(0,len(obj_list),read_obj_num):
        #choose font image---------------------------------------
        fontlist = [cv2.imread(os.path.join(src_path, obj_list[j] + '.png'),-1) for j in range(i, i+read_obj_num)]
        #choose to scale or not----------------------------------
        # src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # src_new = img
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1) 
        #set all the write path-----------------------------
        # res_set_name = img_item[:-4]+'_'+ name
        res_set_name = list1_[l_num]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs---------------------------------------------
        pin_big_small(fontlist,dst_new,al_set,res_set,fraction)

def batch_2_distance(src_path, obj_list, dst_path, res_path, gt_path,name,read_obj_num,fraction_part):
    list1 = [];#list里存入背景图的完整路径
    list1_= [];#存背景图的名称
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)
        list1_.append(dst_item[:-4])
    total = len(list1)
    for i in range(0,len(obj_list),read_obj_num):
        #choose font image---------------------------------------
        fontlist = [cv2.imread(os.path.join(src_path, obj_list[j] + '.png'),-1) for j in range(i, i+read_obj_num)]
        #choose to scale or not----------------------------------
        # src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # src_new = img
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)
        #set all the write path-----------------------------
        # res_set_name = img_item[:-4]+'_'+ name
        res_set_name = obj_list[i] +'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs---------------------------------------------
        pin_distance(fontlist,dst_new,al_set,res_set,fraction_part)



#-----------------------------------------------------------------------------------------------

#---------------------------------------
data_root = '/data1/liumengmeng/_data_CG/'
#--------------------------------------
obj_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')

# #-------多个相同object的图--------------
# obj_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')
# n = 'm1'#新的合成图要存放的文件夹
# name = 'm1_' #新的合成图的命名
# obj_num = 6 #obj的数量
# bg_part = 6 #背景图分为几部分
# gene_num = 100 #随机挑的object数量 = 最后生成的合成图数量
# obj_list = get_obj_sample_list(obj_list, gene_num)
# src_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
# paths = makepath(n, data_root)
# makefolder(paths)
# batch_new(src_path, obj_list, dst_path, paths[0], paths[1], name, obj_num,bg_part)


# #-------多个小物体-----------------------------------------------------------
# n = 'm3'#新的合成图要存放的文件夹
# name = 'm3_6' #新的合成图的命名
# bg_part = 6 #背景图分为几部分
# pic_num = 100 #需要此类的图的数量
# gene_num = bg_part*pic_num #计算所需的object数量
# obj_list = get_obj_sample_list(obj_list, gene_num)
# src_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
# paths = makepath(n, data_root)
# makefolder(paths)
# batch_smalls(src_path, obj_list, dst_path, paths[0], paths[1], name, bg_part)


# #---------一个大物体一个小物体 按背景resize--------------------------------------------
# n = 'm4_test'#新的合成图要存放的文件夹
# name = 'm4_test' #新的合成图的命名
# read_obj_num = 2 #一次读入几个object
# pic_num = 30 #需要此类的图的数量
# fraction = 1/2 #大物体所占背景空间的比例
# gene_num = read_obj_num*pic_num #计算所需的object数量
# obj_list = get_obj_sample_list(obj_list, gene_num)
# src_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
# paths = makepath(n, data_root)
# makefolder(paths)
# batch_2_big_small(src_path, obj_list, dst_path, paths[0], paths[1],name,read_obj_num,fraction)


#--------------两个物体各占两边 1/4  |  各占一半   1/2-------------------------------------------
n = 'm4'#新的合成图要存放的文件夹
name = 'm4_2' #新的合成图的命名
read_obj_num = 2 #一次读入几个object
pic_num = 500 #需要此类的图的数量
fraction_part = 2 #空间分成几份
gene_num = read_obj_num*pic_num #计算所需的object数量
obj_list = get_obj_sample_list(obj_list, gene_num)
src_path = data_root + '_a_obj_all/'
dst_path = data_root + '_bg_all/'
paths = makepath(n, data_root)
makefolder(paths)
batch_2_distance(src_path, obj_list, dst_path, paths[0], paths[1],name,read_obj_num,fraction_part)












#test-----------------------------------------------------------------
# fg = cv2.imread(data_root+"_a_src_full/COCO_train2014_000000000332.png",-1)
# bg = cv2.imread(data_root+"_a_dst/bing_bg_1_0105.jpg")
# ins = cv2.imread(data_root+"_a_ins/COCO_train2014_000000000853.png",-1)
# gt_4667 = cv2.imread(data_root+"_a_gt_full/COCO_train2014_000000000110_3.png",-1)
