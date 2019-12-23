import numpy as np
import pylab
import pandas as pd
 
#define sigmoid function
def sigmoid(x):
    return 1.716 * np.tanh((x * 0.667))
def d_sigmoid(x):
    return 1.716 * 0.667 * (1 - np.square(np.tanh(x*0.667)))

X = np.array([[0.28, 1.31, -6.2],
    [0.07, 0.58, -0.78],
    [1.54, 2.01, -1.63],
    [-0.44, 1.18, -4.32],
    [-0.81, 0.21, 5.73],
    [1.52, 3.16, 2.77],
    [2.20, 2.42, -0.19 ],
    [0.91, 1.94, 6.21],
    [0.65, 1.93, 4.38],
    [-0.26, 0.82, -0.96]])
Y = np.array([[0.011, 1.03, -0.21],
    [1.27, 1.28, 0.08],
    [0.13, 3.12, 0.16],
    [-0.21, 1.23, -0.11],
    [-2.18, 1.39, -0.19],
    [0.34, 1.96, -0.16],
    [-1.38, 0.94, 0.45],
    [-0.12, 0.82, 0.17],
    [-1.44, 2.31, 0.14],
    [0.26, 1.94, 0.08]])

# # 读取数据
# X = pd.read_csv("/Users/Cheney/Downloads/X.csv").values
# Y = pd.read_csv("/Users/Cheney/Downloads/Y.csv").values
 
 
# 初始化输入层到隐藏层权值
s0 = 1 - 2 * np.random.random((3,1))
# 初始化隐藏层到输出层权值
s1 = 1 - 2 * np.random.random((1,1))
#s0 = np.array([0.5, 0.5, 0.5])  #第二小问s0全部为0.5,s1为-0.5
#s1 = np.array([-0.5])
η= 0.01 # 学习率
b1 = 0  # 初始化偏置
b2 = 0
error = [] # 误差
 
for i in range(800):
    j = np.random.randint(0,10) #随机抽取样本
    l0 = np.array(X[j]) # 输入层
    y = np.array(Y[j])  # 输出层
    net1 = b1 + s0[0]*l0[0] + s0[1]*l0[1] +s0[2]*l0[2] # 输入层到隐藏层
    l1 = sigmoid(net1)
    net2 = b2 + l1*s1[0] # 隐藏层到输出层
    l2 = sigmoid(net2)
    # print(l2)
    g2 = (y - l2) * d_sigmoid(l2) # 反向传播，计算输出层神经元的梯度
    # print(d_sigmoid(l2))
    g1 = g2 * s1[0] * d_sigmoid(l1)  # 反向传播，计算隐藏层神经元的梯度
    b1 = b1 + η * g2 # 反向传播，更新隐藏层权重
    s1[0] =  s1[0] + η * l1 * g2
    b1 += η * g1  # 反向传播，更新输入层权重
    n = len(s0)
    for m in range(n):
        s0[m] = s0[m] + η * g1 * l0[m]
 
    e = 0.5 * np.square(y - l2)
    # print(e)
    error.append(e)
 
# # 画图
# pylab.plot(range(len(error)), error, 'r-')
# pylab.xlabel('epoch')
# pylab.ylabel('error')
# pylab.show()
