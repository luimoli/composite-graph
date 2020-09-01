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

def his(img1,img2):
    similarity = 0
    for i in range(3):
        H1 = cv2.calcHist([img1], [i], None, [256],[0,256])
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) # 对图片进行归一化处理
        H2 = cv2.calcHist([img2], [i], None, [256],[0,256])
        H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
        curr_cmp = cv2.compareHist(H1, H2,0)
        # print(curr_cmp)
        similarity += curr_cmp
    return similarity / 3

# img1 = cv2.imread('/data1/liumengmeng/_data_CG/test_inter/75dcf4625a02ba6193a38b3841c5ab86f2feea9e.jpg')
# img2 = cv2.imread('/data1/liumengmeng/_data_CG/_bg_all/photo-1551651431-aae9dc0ba2dd.jpg')
# # yes = his(img1,img2)
# # print(yes)

def get_lowcontrast_id(obj_path,obj_list,inter_path,dst_path,file1,file2, num):
    '''
    num是要选择的最高的相似度的图的数量
    '''
    dic = {} #记录一个object对所有背景图遍历的score {背景图id：scor}
    dic_obj_bgid = {} #记录每个object的最好的背景图id
    dic_obj_bgscore = {}#记录每个object的最好的背景图的score
    for i in tqdm(obj_list):
        img_path = obj_path + i + '.png'
        img_obj_path = inter_path + i + '.jpg'
        get_obj_pixel(img_path,img_obj_path)

        img = cv2.imread(img_obj_path)
        for item in os.listdir(dst_path):
            item_path = os.path.join(dst_path,item)
            img1 = cv2.imread(item_path) #读入背景图
            similarity = his(img, img1)
            dic.update({item[:-4] : similarity})
        dic_s = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
        # print(dic_s[0][1])
        dic_obj_bgid.update({i : dic_s[0][0]})
        dic_obj_bgscore.update({i : dic_s[0][1]})
        tqdm.write(f'{i} : {dic_s[0][0]} = {dic_s[0][1]}')

    obj_s = sorted(dic_obj_bgscore.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    # num = len(obj_s)
    for i in range(num): # num是要选择的
        print(obj_s[i][0],file = file1) #排在前面的拥有匹配的较高相似度背景的object 的id
        print(dic_obj_bgid[obj_s[i][0]], file= file2) #与object对应的背景id



def get_obj_sample_list(obj_list,num):
    samplelist = random.sample(obj_list, num)
    return samplelist




inter_path = data_root + 'test_inter/'# #object变形的中间图保存路径
obj_path = data_root + '_a_obj_all/'
dst_path = data_root + '_bg_all/'
obj_list = txt2list('/data1/liumengmeng/_data_CG/id/obj_all.txt')
#----------------------------------------------------------------
# #选择一些objects
gene_num = 300
num = 200 #从num中选择数值较高的前num个
obj_list = get_obj_sample_list(obj_list, gene_num)


txtname = 'obj_'+str(num)
list2txt(obj_list,'/data1/liumengmeng/_data_CG/id/' + txtname + '.txt')
file1 = open(data_root+'id/contrast_'+ txtname + '_obj.txt', 'w')
file2 = open(data_root+'id/contrast_'+ txtname + '_bg.txt', 'w')

get_lowcontrast_id(obj_path,obj_list,inter_path,dst_path,file1,file2, num)








#-------------------singel test--------------------------------------------------------------------------------------------
# dic = {}
# for item in os.listdir(bgpath):
#     item_path = os.path.join(bgpath,item)
#     img1 = cv2.imread(item_path)
#     img = cv2.imread('/data1/liumengmeng/_data_CG/test_inter/74044521530edeb187b765f2a3c2b880b8f8859f.jpg')
#     # img1 = cv2.imread('/data1/liumengmeng/_data_CG/_bg_5207/photo-1547657126-fda03ca12d7a.jpg')
#     # 计算图img的直方图
#     H1 = cv2.calcHist([img], [1], None, [256],[0,256])
#     H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1) # 对图片进行归一化处理
    
#     # 计算图img2的直方图
#     H2 = cv2.calcHist([img1], [1], None, [256],[0,256])
#     H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)

#     # 利用compareHist（）进行比较相似度
#     similarity = cv2.compareHist(H1, H2,0)
#     dic.update({item[:-4] : similarity})
# d3 = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
# print(d3[0])



