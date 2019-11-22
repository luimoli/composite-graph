import os
import cv2
import numpy as np
from PIL import Image
import random
import shutil
import json

def rotate(image):#旋转缩放一定角度
    (h, w) = image.shape[:2]
    angle = random.randint(-8,8)
    scale = random.uniform(0.75,1.0)
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

def resize1(font,back,ins):#得到按照背景大小缩放后的f_rs
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
    return f_new,alpha_255

#-------确定放在背景图上的位置-------------------------------------------------
def get_bnew(f_rs,back,a,b,c,d,alpha_path):#决定放在背景图的位置 左上角
    #r1, c1 = shape(f_rs)
    b_new = back[a:b,c:d]
    b_new = b_new.astype(float)

    alpha_full = np.zeros(back.shape, dtype=back.dtype)
    #ins_full = np.zeros(back.shape, dtype=back.dtype)
    #alpha_full = cv2.imread(old_a_pth)#读入上一张完整图片的alpha图
    alpha_full[a:b,c:d] += get_fnew_and_alpha(f_rs)[1]
    #ins_full[a:b,c:d] += ins
    cv2.imwrite(alpha_path, alpha_full)
    #cv2.imwrite(color_path, ins_full)
    return b_new

def get_bnew1(f_rs,back,a,b,c,d,alpha_path,old_a_pth):#决定放在背景图的位置 左上角
    #r1, c1 = shape(f_rs)
    b_new = back[a:b,c:d]
    b_new = b_new.astype(float)

    #alpha_full = np.zeros(back.shape, dtype=back.dtype)
    alpha_full = cv2.imread(old_a_pth)#读入上一张完整图片的alpha图
    alpha_full[a:b,c:d] += get_fnew_and_alpha(f_rs)[1]
    cv2.imwrite(alpha_path, alpha_full)
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
    f_new,alpha_255 = get_fnew_and_alpha(f_rs)
    alpha = alpha_255.astype(float)/255#把alpha/255得到的0和1结果用于之后的权数
    f_new = cv2.multiply(alpha,f_new)
    b_new = cv2.multiply(1-alpha,b_new)
    outImage = f_new + b_new
    return outImage

#------------------------------------------------------------------
def pin(font,back,alpha_set,res_set):#合成在左上角
    f_rs = resize(font,back)
    #f_rs, i_rs = resize1(font,back,ins)
    #f_rs = left(f)
    r1, c1 = shape(f_rs)
    b_r,b_c = shape(back)
    a,b,c,d = 0,r1,(b_c-c1),b_c#--------右上 ru
    b_new = get_bnew(f_rs,back,a,b,c,d,alpha_set)
    outImage = conver(f_rs,b_new)
    res = back.copy()
    res[a:b,c:d] = outImage
    cv2.imwrite(res_set, res)

def pin1(font,back,alpha_path,old_a_pth,res_set):#合成在右下角
    f_rs = resize(font,back)
    r1, c1 = shape(f_rs)
    b_r,b_c = shape(back)
    #a,b,c,d = 0,r1,0,c1#----------------左上 lu
    #a,b,c,d = (b_r-r1),b_r,0,c1#--------左下 ld
    #a,b,c,d = 0,r1,(b_c-c1),b_c#--------右上 ru	
    a,b,c,d = (b_r-r1),b_r,(b_c-c1),b_c#右下 rd
    b_new = get_bnew1(f_rs,back,a,b,c,d,alpha_path,old_a_pth)
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

def gene_dict_new(img_name,back_name,json_old_path,dict_new_path):
    dic = {}
    dicdata = get_json(json_old_path)
    dic.update({"background":back_name})
    dic_content= {'type' : dicdata[img_name], 'url' : img_name}
    #dic_content= {dicdata[img_name] : img_name}
    dic.update({'1':dic_content})
    with open(dict_new_path,"w",encoding="utf-8") as f:
        f.write(json.dumps(dic))

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
        #src_new = img

        #choose txt-----------------------------------------
        #txt_path = os.path.join(name_path, img_item[:-4]+'.txt')
        json_old_path = os.path.join(name_path, 'instance_json_new.txt')
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
        pin(src_new,dst_new,al_set,res_set)#存储合成图并存储了gt
        #gene_dict(img_item, list1_[l_num] , txt_path , dic_set)
        gene_dict_new(img_item[:-4] ,list1_[l_num],json_old_path,dic_set)

def batch1(src_path, dst_path, res_path, gt_path,gt_old_path,name_path,dict_path,dict_old_path,name):
    list1 = [];
    list2 = [];
    for dst_item in os.listdir(dst_path):
        back_path = os.path.join(dst_path, dst_item)
        list1.append(back_path)#存背景图的完成路径
        list2.append(dst_item)#存背景图的名称
    total = len(list1)
    for img_item in os.listdir(src_path):
        img_path = os.path.join(src_path, img_item)
        img = cv2.imread(img_path,-1)
        src_new = rotate(img)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #src_new = img

        txt_path = os.path.join(name_path, img_item[:-4]+'.txt')#获取src图的txt文件的路径

        l_num = random.randint(0,total-1)
        dst_new = cv2.imread(list1[l_num],-1)#选取一张随机的背景图(合成图)

        res_set_name = img_item[:-4]+'_'+ name
        al_set = os.path.join(gt_path, res_set_name +'.png')#设定存储gt的完整路径和命名
        dic_set = os.path.join(dict_path, res_set_name +'.txt')#设定存储dicts的完整路径和命名
        res_set = os.path.join(res_path, res_set_name +'.jpg')
        old_dic_path = os.path.join(dict_old_path,list2[l_num][:-4]+'.txt')#找到此背景图的json信息
        old_a_path = os.path.join(gt_old_path, list2[l_num][:-4]+'.png')#找到此背景图的gt(已经是合成图)
        pin1(src_new, dst_new, al_set, old_a_path, res_set)#生成新合成图并存储了新的gt
        gene_dict1(img_item,txt_path,dic_set,old_dic_path)



#-----------------------------------------------------------------------------------------------
def makefolder(path):
    if os.path.exists(path):
        print(path + "  is existed")
    else:
        os.makedirs(path)
        print(path + "  is created!")
    #if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    
#---------------------------------------
data_root = '/data0/liumengmeng/data/'
#--------------------------------------
n = 'b'#新的合成图要存放的文件夹
name = n #新的合成图的命名
src_path = data_root + '_a_src_full/'
dst_path = data_root + '_a_dst/'
name_path = data_root + 'id/'
#ins_path = '_a_ins'

res_path = data_root + 'pic_res/'+ n +'/'
gt_path = data_root + 'pic_gt/'+ n + '/'
dict_path = data_root + 'pic_dic/' + n + '/'
#color_path = 'pic_ins/' + n + '/'
makefolder(res_path)
makefolder(gt_path)
makefolder(dict_path)
# makefolder(color_path)
# batch(src_path, dst_path, res_path, gt_path,name_path,dict_path,ins_path,color_path,name)
batch(src_path, dst_path, res_path, gt_path,name_path,dict_path,name)