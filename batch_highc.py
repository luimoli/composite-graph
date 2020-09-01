import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil
import json
from utils.func import *

def shape(pic):
    r,c = pic.shape[0:2]
    return r,c

def resize_big(font,back):#得到按照背景大小缩放后的f_rs
    # from ipdb import set_trace;set_trace()
    f_r,f_c = shape(font)
    b_r,b_c = shape(back)
    k = max(b_c/f_c, b_r/f_r)
    f_rs_ = cv2.resize(font,None,fx=k,fy=k, interpolation = cv2.INTER_CUBIC)
    f_r_new,f_c_new = shape(f_rs_)
    r_ = (f_r_new - b_r) // 2
    c_ = (f_c_new - b_c) // 2
    a,b,c,d = r_, r_+b_r, c_, c_+b_c
    f_rs = f_rs_[a:b,c:d].copy()
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


#-------得到前景RGB图和alpha图------------------------------------------------
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

    alpha_pic = alpha_full.copy()
    af = alpha_pic[a:b,c:d].copy()
    alpha_pic[a:b,c:d] = cv2.add(af,get_fnew_and_alpha(f_rs)[1])
    # cv2.imwrite(alpha_path, alpha_full)#---------存储gt
    return b_new, alpha_pic #根据font的bg切片  /  新的gt

#-------合成-------------------------------------------------
def conver(f_rs,b_new):#在背景图上取一块与缩放后的前景图相同的区域与f_rs融合得到f_rs大小的融合图
    f_new,alpha_255,alpha_ori = get_fnew_and_alpha(f_rs)
    alpha = alpha_ori.astype(float)/255#把alpha/255得到的0和1结果用于之后的权数
    f_new = cv2.multiply(alpha,f_new)
    b_new = cv2.multiply(1-alpha,b_new)
    outImage = f_new + b_new
    return outImage

#-------choose the position of objects------------------------------------------------
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

def choose_pos_random_xxyy(fp,a,b,c,d): #根据给定好的坐标来选择在其范围内的坐标
    print('this is random!')
    # from ipdb import set_trace;set_trace()
    r1, c1 = shape(fp)
    xr = random.randint(a, b-r1)
    xc = random.randint(c,d-c1)
    size = [xr,xr+r1,xc,xc+c1]
    return size

def choose_full_ori(fp, bp):
    f_r, f_c = shape(fp)
    b_r, b_c = shape(bp)
    size = []
    rnum = int(b_r / f_r + 1)
    cnum = int(b_c / f_c + 1)
    rtmp = int((b_r-f_r)/(rnum-1))
    ctmp = int((b_c-f_c)/(cnum-1))
    for i in range(rnum):
        for j in range(cnum):
            size.append([rtmp*i,rtmp*i+f_r,ctmp*j,ctmp*j+f_c])
    # from ipdb import set_trace; set_trace()
    return size,rnum,cnum

def choose_full(fp, bp, dense_k):
    f_r, f_c = shape(fp)
    b_r, b_c = shape(bp)
    size = []
    rnum = int(b_r / f_r + dense_k)
    cnum = int(b_c / f_c + dense_k)
    rtmp = int((b_r-f_r)/(rnum-1))
    ctmp = int((b_c-f_c)/(cnum-1))
    for i in range(rnum):
        for j in range(cnum):
            size.append([rtmp*i,rtmp*i+f_r,ctmp*j,ctmp*j+f_c])
    # from ipdb import set_trace; set_trace()
    return size,rnum,cnum

#------------------------------------------------------------------------------------------------------------------
def pin_high_contrast_1(fontlist,back,alpha_set,res_set,bg_part):
    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    f_rs_b = resize_big(fontlist[0], back)
    f_rs_s = resize_part(fontlist[1], back, bg_part)[0]
    size_b = choose_pos_center(f_rs_b,back)
    size_s = choose_pos_random_xxyy(f_rs_s,*size_b) #在大物体的位置范围内随机选择一个位置，保证小物体在大物体上

    b_new = get_bnew(f_rs_b,back,*size_b,alpha_full)[0] #合成较大的物体图，但是不认为显著不保存alpha图
    outImage = conver(f_rs_b,b_new)
    res = back.copy()
    res[size_b[0]:size_b[1],size_b[2]:size_b[3]] = outImage
    
    back = res.copy()

    b_new,alpha_new = get_bnew(f_rs_s,back,*size_s,alpha_full)
    outImage = conver(f_rs_s,b_new)
    res = back.copy()
    res[size_s[0]:size_s[1],size_s[2]:size_s[3]] = outImage

    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)

def pin_full(font,font_up,back,alpha_set,res_set,bg_part,font_bg_part, dense_k):#多个相同object平铺铺满
    f_rs,flag = resize_part(font,back,bg_part)
    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    back = back.copy()
    size,rnum,cnum = choose_full(f_rs, back, dense_k)
    for i in range(rnum*cnum):
        a,b,c,d =  size[i]
        b_new = get_bnew(f_rs,back,a,b,c,d,alpha_full)[0]
        outImage = conver(f_rs,b_new)
        res = back.copy()
        res[a:b,c:d] = outImage
        back = res
    f_rs_up,flag_up = resize_part(font_up,back,font_bg_part)
    pos = choose_pos_random(f_rs_up,back)
    b_new,alpha_new = get_bnew(f_rs_up,back,*pos,alpha_full)
    outImage = conver(f_rs_up,b_new)
    res = back.copy()
    res[pos[0]:pos[1],pos[2]:pos[3]] = outImage

    cv2.imwrite(res_set, res)
    cv2.imwrite(alpha_set, alpha_new)


#-------------------------------------------------------------------

def batch_high_contrast_1(src_path, obj_list, dst_path, res_path, gt_path,name,read_obj_num,bg_part):
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
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)
        #set all the write path-----------------------------
        res_set_name = obj_list[i] +'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs---------------------------------------------
        pin_high_contrast_1(fontlist,dst_new,al_set,res_set,bg_part)

def batch_high_contrast_2(src_path, obj_list, dst_path, res_path, gt_path,name,read_obj_num,bg_part,font_bg_part,dense_k):
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
        #choose back randomly----------------------------------------
        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)
        #set all the write path-----------------------------
        res_set_name = obj_list[i] +'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        #funcs---------------------------------------------
        pin_full(fontlist[0],fontlist[1],dst_new,al_set,res_set,bg_part,font_bg_part,dense_k)



#-----------------------------------------------------------------------------------------------

#---------------------------------------
data_root = '/data1/liumengmeng/_data_CG/'
#--------------------------------------
obj_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')


#--------------一个大物体铺满/一个小物体在大物体上-------------------------------------------
n = 'h1_batch2'#新的合成图要存放的文件夹
name = 'h1_batch2' #新的合成图的命名
read_obj_num = 2 #一次读入几个object
pic_num = 3000 #需要此类的图的数量
bg_part = 3 #小物体的大小占背景图的比例
gene_num = read_obj_num*pic_num #计算所需的object数量
obj_list = get_obj_sample_list(obj_list, gene_num)
src_path = data_root + '_a_obj_all/'
dst_path = data_root + '_bg_all/'
paths = makepath(n, data_root)
makefolder(paths)
batch_high_contrast_1(src_path, obj_list, dst_path, paths[0], paths[1],name,read_obj_num,bg_part)



# n = 'h2_3'#新的合成图要存放的文件夹
# name = 'h2_3' #新的合成图的命名
# read_obj_num = 2 #一次读入几个object
# pic_num = 800 #需要此类的图的数量
# bg_part = 4 #小物体的大小占背景图的比例
# font_bg_part = 3 #放在上面的物体所占背景的大小
# dense_k = 3 #调节object平铺的密集程度
# gene_num = read_obj_num*pic_num #计算所需的object数量
# obj_list = get_obj_sample_list(obj_list, gene_num)
# src_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
# paths = makepath(n, data_root)
# makefolder(paths)
# batch_high_contrast_2(src_path, obj_list, dst_path, paths[0], paths[1],name,read_obj_num,bg_part,font_bg_part,dense_k)









#test-----------------------------------------------------------------
# fg = cv2.imread(data_root+"_a_src_full/COCO_train2014_000000000332.png",-1)
# bg = cv2.imread(data_root+"_a_dst/bing_bg_1_0105.jpg")
# ins = cv2.imread(data_root+"_a_ins/COCO_train2014_000000000853.png",-1)
# gt_4667 = cv2.imread(data_root+"_a_gt_full/COCO_train2014_000000000110_3.png",-1)
