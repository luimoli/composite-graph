import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import math
import torch
from utils.func import *

class ImageData(data.Dataset):
    def __init__(self, img_root, label_root, transform, t_transform,  filename=None):
        if filename is None:
            print('filename is none!')
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
            self.label_path = list(
                map(lambda x: os.path.join(label_root, x.split('/')[-1][:-3] + 'png'), self.image_path))
        else:
            lines = [line.rstrip('\n') for line in open(filename)]
            # print(len(lines))
            self.image_path = list(map(lambda x: os.path.join(img_root, x + '.png'), lines))
            self.label_path = list(map(lambda x: os.path.join(label_root, x + '.png'), lines))
        
        self.transform = transform
        self.t_transform = t_transform

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('L')
        label = Image.open(self.label_path[item]).convert('L')
        if self.transform is not None:
            image = self.transform(image)
        if self.t_transform is not None:
            label = self.t_transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)


def eval_mae(y_pred, y):
    return torch.abs(y_pred - y).mean()



def get_10per_lowmae(dataset,device,txtpath):
    avg_mae = 0.0
    dic = {}
    for i, data_batch in enumerate(dataset):
        images, labels = data_batch
        images = images.unsqueeze(0)
        labels = labels.unsqueeze(0)
        images, labels = images.to(device), labels.to(device)
        print(dataset.label_path[i])
        print(dataset.label_path[i][45:-4])
        imae = eval_mae(images, labels).item()
        avg_mae += imae
        dic.update({imae: dataset.label_path[i][45:-4]})################################

    dicsort=sorted(dic.keys(),reverse=True)
    for i in range(int(len(dicsort)*0.1)):
        print(dic[dicsort[i]],file=txtpath)
    return avg_mae / len(dataset)


#----------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])
t_transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device('cuda:0')

# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/DUTS-OMRON_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/DUT-OMRON/imgs'

# img_root = '/data1/liumengmeng/dataset/BASNet/DUTS-TE'
# label_root = '/data1/liumengmeng/dataset/BASNet_gt/DUTS-TE'
# id_path = '/data1/liumengmeng/dataset/BASNet_id/DUTS-TE.txt'
# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/DUTS-TE_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/DUTS/imgs'

# img_root = '/data1/liumengmeng/dataset/BASNet/ECSSD'
# label_root = '/data1/liumengmeng/dataset/BASNet_gt/ECSSD'
# id_path = '/data1/liumengmeng/dataset/BASNet_id/ECSSD.txt'
# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/ECSSD_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/ECSSD/imgs'

# img_root = '/data1/liumengmeng/dataset/BASNet/HKU-IS'
# label_root = '/data1/liumengmeng/dataset/HKU-IS/gt'
# id_path = '/data1/liumengmeng/dataset/BASNet_id/HKU-IS.txt'
# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/HKU-IS_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/HKU-IS/imgs'

# img_root = '/data1/liumengmeng/dataset/BASNet/PASCAL-S'
# label_root = '/data1/liumengmeng/dataset/PASCAL-S/gt'
# id_path = '/data1/liumengmeng/dataset/BASNet_id/PASCAL-S.txt'
# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/PASCAL-S_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/PASCAL-S/imgs'

# img_root = '/data1/liumengmeng/dataset/BASNet/SOD'
# label_root = '/data1/liumengmeng/dataset/BASNet_gt/SOD/gt'
# id_path = '/data1/liumengmeng/dataset/BASNet_id/SOD.txt'
# id_mae_path = '/data1/liumengmeng/dataset/BASNet_id/SOD_mae.txt'
# rgb_root = '/data1/liumengmeng/dataset/SOD/imgs'


# img_root = '/data1/liumengmeng/dataset/DUTS/imgs'
# id_path = '/data1/liumengmeng/dataset/DUTS/ImageSets/total_id.txt'
# # ----获得数据集的gt的id以便后续读取-------------
# id_path1 = open(id_path,'w')
# getid(img_root,id_path1)


# # ---构建一个imgdata类来同时处理predicted map和gt---------------
# file = id_path
# txtpath = open(id_mae_path,'w')
# dataset = ImageData(img_root,label_root, transform, t_transform,filename=file)
# print(len(dataset))
# # ---进行mae计算并将mae最高的10%的图片的id写入txt文件
# print(get_10per_lowmae(dataset,device,txtpath))



# tofolder = '/data1/liumengmeng/dataset/BASNet_mae/SOD_map'
# copy_by_idtxt(id_mae_path, img_root, tofolder)
# tofolder = '/data1/liumengmeng/dataset/BASNet_mae/SOD_gt'
# copy_by_idtxt(id_mae_path, label_root, tofolder)
# tofolder = '/data1/liumengmeng/dataset/BASNet_mae/SOD_rgb'
# copy_by_idtxt(id_mae_path, rgb_root, tofolder)
