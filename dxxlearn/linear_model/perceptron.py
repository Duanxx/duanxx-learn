#-*- coding=utf-8 -*-
"""
@file ： perceptron.py
@author : duanxxnj@163.com
@time : 2016/11/5 19:56
"""
import time
import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


"""
    感知机算法实现
"""
class Perceptron(object):
    """
        n_iter 是感知机算法的迭代次数
        eta 是感知机算法权值更新的系数，这个系数越小，更新步长越小
    """
    def __init__(self, n_iter=1, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta

    """
        感知机算法学习函数，这个函数学习感知机的权值向量W
        X   [n_samples, n_features]二维向量，数据样本集合，其第一列全部为1
        y   [n_samples]以为向量，数据的标签，这里为[+1,-1]
        fun 感知机所使用的优化函数：“batch”：批处理梯度下降法
                                “SGD”：随机梯度下降法
        isshow 是否将中间过程绘图出来：False ：不绘图
                                   True : 绘图
    """
    def fit(self, X, y, fun="batch",isshow=False):
        n_samples, n_features = X.shape #获得数据样本的大小
        self.w = np.ones(n_features, dtype=np.float64) #参数W

        if isshow == True:
            plt.ion()
        # 如果是批处理梯度下降法
        if fun == "batch":
            for t in range(self.n_iter):
                error_counter = 0
                # 对分类错误的样本点集合做权值跟新更新
                for i in range(n_samples):
                    if self.predict(X[i])[0] != y[i]:
                        # 权值跟新系数为 eta/n_iter
                        self.w += y[i] * X[i] * self.eta/self.n_iter
                        error_counter = error_counter +1
                if (isshow):
                    self.plot_process(X)
                # 如果说分类错误的样本点的个数为0，说明模型已经训练好了
                if error_counter == 0:
                    break;
        # 如果是随机梯度下降法
        elif fun == "SGD":
            for t in range(self.n_iter):
                # 每次随机选取一个样本来更新权值
                i = random.randint(0, n_samples-1)
                if self.predict(X[i])[0] != y[i]:
                    # 为了方便这里的权值系数直接使用的是eta而不是eta/n_iter
                    # 并不影响结果
                    self.w += y[i] * X[i] * self.eta
                if (t%5 == 0 and isshow):
                    self.plot_process(X)
                # 此处本应该加上判断模型是否训练完成的代码，但无所谓

    """
        预测样本类别
        X   [n_samples, n_features]二维向量，数据样本集合，其第一列全部为1
        return 样本类别[+1, -1]
    """
    def predict(self, X):
        X = np.atleast_2d(X)#如果是一维向量，转换为二维向量
        return np.sign(np.dot(X, self.w))

    """
        绘图函数
    """
    def plot_process(self, X):
        fig = plt.figure(1)
        fig.clear()
        # 绘制样本点分布
        plt.scatter(X[0:50, 1], X[0:50, 2], c='r')
        plt.scatter(X[50:100, 1], X[50:100, 2], c='b')
        # 绘制决策面
        xx = np.arange(X[:, 1].min(), X[:, 1].max(), 0.1)
        yy = -(xx * self.w[1] / self.w[2]) - self.w[0] / self.w[2]
        plt.plot(xx, yy)

        plt.grid()
        plt.pause(1)


if __name__ == '__main__':
    # 加载数据
    iris_data = load_iris()
    y = np.sign(iris_data.target[0:100] - 0.5)
    X = iris_data.data[0:100, [0, 3]]
    X = np.c_[(np.array([1] * X.shape[0])).T, X]

    # 选择算法输入1/2
    choose = input("choose batch or SGD(1/2):")
    if choose == 1:
        clf = Perceptron(30, 0.03)
        clf.fit(X, y, "batch", True)
    elif choose == 2:
        clf = Perceptron(200, 0.1)
        clf.fit(X, y, "SGD", True)
