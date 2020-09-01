import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from utils.func import *
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

class TargetData(data.Dataset):
    """ image dataset
    as ImageData without labels
    """

    def __init__(self, img_root, transform, filename=None):
        if filename is None:
            self.image_path = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
        else:
            lines = [line.rstrip('\n') for line in open(filename)]
            # lines = lines * int(np.ceil(float(max_iters) / len(lines)))  #only extend ids when using _next_()
            self.image_path = list(map(lambda x: os.path.join(img_root, x + '.jpg'), lines))

        self.transform = transform

    def __getitem__(self, item):
        try:
            image = Image.open(self.image_path[item]).convert('L')
            if self.transform is not None:
                image = self.transform(image)
            return image
        except IOError as ercode:
            print(self.image_path[item],file=filebad_id)
        except TypeError as tycode:
            print(self.image_path[item],file=filebad_id)

    def __len__(self):
        return len(self.image_path)

img_root = '/data1/liumengmeng/objects'

# dataset = TargetData(img_root,None,None)
# dic = {}
# for i, img in enumerate(dataset):
#     # print(dataset.image_path[i][img_root.find('lts')+4 :-4])
#     imga = np.array(img)
#     num = imga.shape[0] * imga.shape[1]
#     index = np.argwhere(imga  == 0)
#     ratio = (num - len(index)) / num
#     dic.update({dataset.image_path[i][img_root.find('cts')+4 :-4] : ratio})
#     # print(img)

# dic_sorted = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]))
# file1 = open('/data1/liumengmeng/obj_results_rec.txt','w')
# file2 = open('/data1/liumengmeng/obj_results_id.txt','w')
# for i in dic_sorted:
#     print(f'{i[0]} : {i[1]}',file = file1)
#     print(i[0],file = file2)



#---------选id并且复制挑出来----------------------------------------------------
# idlist = txt2list('/data1/liumengmeng/obj_results_id.txt')
# idn = []
# for i in range(670):
#     idn.append(idlist[i])
# list2txt(idn,'/data1/liumengmeng/obj_results_id_20.txt')

# copy_by_idtxt('/data1/liumengmeng/obj_results_id_670.txt','/data1/liumengmeng/objects','/data1/liumengmeng/objects_670','.png')


idlist = txt2list('/data1/liumengmeng/test_DUTS_id.txt')
idn = []
for i in range(5017):
    if i >= 4515:
        idn.append(idlist[i])
list2txt(idn,'/data1/liumengmeng/test_DUTS_id_mae.txt')

# copy_by_idtxt('/data1/liumengmeng/test_DUTS_id_mae.txt','/data1/liumengmeng/bg_16231','/data1/liumengmeng/_data_CG/_bg_all','.jpg')



#----------------单个测试------------------------
# img = (dataset[0])
# img.save('img.png')
# imga = np.array(img)
# num = imga.shape[0] * imga.shape[1]
# index = np.argwhere(imga == 0)
# print(imga.shape,len(index))