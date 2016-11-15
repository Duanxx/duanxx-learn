#-*- coding=utf-8 -*-
"""
@file ： test_perceptron.py
@author : duanxxnj@163.com
@time : 2016-11-15
"""

import matplotlib.pyplot as plt
import numpy as np
from dxxlearn.linear_model.perceptron import Perceptron

# 这里使用单个感知机，单输入做为测试样本
# 主要用于测试初始化权值对平方误差损失函数收敛速率的影响
clf = Perceptron(300, 0.01)
# 初始化权值为w=[2, 2]
cost1 = clf.train(np.array([[1, 1]]), [0], clf.sigmoid, w_in=2)
# 初始化权值为w=[0.6, 0.6]
cost2 = clf.train(np.array([[1, 1]]), [0], clf.sigmoid, w_in=0.6)
plt.plot(cost1, label='init w=[2, 2]', c='b')
plt.plot(cost2, label='init w=[0.6, 0.6]', c='r')
plt.legend()
plt.grid()
plt.show()
