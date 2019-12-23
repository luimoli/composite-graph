import os
import math
# import cv2
import numpy as np
from PIL import Image
import random
import json
from random import randint
import shutil
import math
import torch as t
import torch.nn.functional as F
from tqdm import tqdm
data_root = '/data0/liumengmeng/data/'

# a = [1,2,3,4,5,7]
# print(a.index(7))

A=np.array([[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8]])
# print(A.shape)
A= t.from_numpy(A)
# print(A.shape)
# print(A.dim())


c = np.random.random((2,1,3,4))
c= t.from_numpy(c)
c1 = F.softmax(c, dim=0)

# a = [1,2,3,4,5,6,6,7,7,8,9,0]
# for i in tqdm(a):
#     print(i)

j = np.random.random((1,1))
print(j)
