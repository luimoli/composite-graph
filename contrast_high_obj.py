import matplotlib.pyplot as plt
import cv2
import numpy as py
import math
from utils.func import *
from tqdm import tqdm

data_root = '/data1/liumengmeng/_data_CG/'

def get_obj_pixel(fg_path, fg_new_path):#只生成object的像素图
    fg = cv2.imread(fg_path,-1)
    b,g,r,a = cv2.split(fg)
    # index_a_255 = np.argwhere(a == 255)#分出alpha通道，找到其中值为255的点的位置，即可定位在原图中不透明的点的位置
    # num_a = len(np.argwhere(a == 255))#255的像素点的总数量
    index_a_255 = np.argwhere(a > 0)#二值化alpha图，取>0的点
    num_a = len(index_a_255)#255的像素点的总数量
    rl = math.ceil(math.sqrt(num_a))#把不透明的部分的像素点变为一个正方形新图的size
    obj = np.zeros((rl,rl,3))#创建一个三通道的图
    obj = obj.astype(np.uint8)#改变数据类型为uint8
    num = 0
    for i in obj:
        for j in i:
            if num < num_a:
                index = index_a_255[num]
                j[0] = fg[index[0]][index[1]][0]
                j[1] = fg[index[0]][index[1]][1]
                j[2] = fg[index[0]][index[1]][2]
                num += 1
    cv2.imwrite(fg_new_path, obj)

def get_all_obj_pixel(src_path,inter_path):
    for item in os.listdir(src_path):
        fg_path = os.path.join(src_path,item)
        fg_new_path = os.path.join(inter_path,item[:-4]+'.jpg')
        get_obj_pixel(fg_path, fg_new_path)
        print('ok')


def his(img1,img2):
    similarity = 0
    b1,g1,r1,a1 = cv2.split(img1)
    b2,g2,r2,a2 = cv2.split(img2)
    for i in range(3):
        H1 = cv2.calcHist([img1], [i], a1, [256],[0,256])
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) # 对图片进行归一化处理
        H2 = cv2.calcHist([img2], [i], a2, [256],[0,256])
        H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
        curr_cmp = cv2.compareHist(H1, H2,0)
        # print(curr_cmp)
        similarity += curr_cmp
    return similarity / 3


# img1 = cv2.imread('/data1/liumengmeng/_data_CG/_a_obj_all/0adc69b95e59fb16b69bbd04923500bd933f1cc8.png',-1)
# img2 = cv2.imread('/data1/liumengmeng/_data_CG/_a_obj_all/0adc69b95e59fb16b69bbd04923500bd933f1cc8.png',-1)
# yes = his(img1,img2)
# print(yes)


def get_highcontrast_id(obj_path,obj_list,file1,file2,num):
    '''
    num是要从目前的obj list里选择的最好的图的数量
    '''
    
    dic = {} #记录一个object对所有背景图遍历的score {背景图id：scor}
    dic_obj_bgid = {} #记录每个object的最好的背景图id
    dic_obj_bgscore = {}#记录每个object的最好的背景图的score
    for i in tqdm(obj_list):
        img_obj_path = obj_path + i + '.png'
        img = cv2.imread(img_obj_path,-1)
        for item in os.listdir(obj_path):
            item_path = os.path.join(obj_path,item)
            img1 = cv2.imread(item_path,-1) #读入背景图
            # print(img.shape,img1.shape)
            similarity = his(img, img1)
            dic.update({item[:-4] : similarity})
        dic_s = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=False)
        # print(dic_s[0][1])
        dic_obj_bgid.update({i : dic_s[0][0]})
        dic_obj_bgscore.update({i : dic_s[0][1]})
        tqdm.write(f'{i} : {dic_s[0][0]} = {dic_s[0][1]}')

    obj_s = sorted(dic_obj_bgscore.items(), key = lambda kv:(kv[1], kv[0]),reverse=False)
    # num = len(obj_s)
    for i in range(num): 
        print(obj_s[i][0],file = file1) #排在前面的object 的id
        print(dic_obj_bgid[obj_s[i][0]], file= file2)



def get_obj_sample_list(obj_list,num):
    samplelist = random.sample(obj_list, num)
    return samplelist




obj_path = data_root + '_a_obj_all/'
# dst_path = data_root + '_bg_all/'
obj_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')
# #----------------------------------------------------------------
# #选择一些objects
gene_num = 400
num = 301 #从num中选择数值较高的前num个
obj_list = get_obj_sample_list(obj_list, gene_num)

txtname = 'obj_highc_'+str(num)
list2txt(obj_list,'/data1/liumengmeng/_data_CG/id/' + txtname + '.txt')
file1 = open(data_root+'id/contrast_'+ txtname + '_obj1.txt', 'w')
file2 = open(data_root+'id/contrast_'+ txtname + '_obj2.txt', 'w')

get_highcontrast_id(obj_path,obj_list,file1,file2, num)



# 先生成所有object图的中间图
# get_all_obj_pixel(obj_path,inter_path)




