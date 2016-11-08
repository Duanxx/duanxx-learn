#-*- coding=utf-8 -*-
"""
@file ： test_sklearn_perceptron.py
@author : duanxxnj@163.com
@time : 2016/11/5 20:53
"""


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_raises

from sklearn.utils import check_random_state
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
random_state = check_random_state(12)
indices = np.arange(iris.data.shape[0])
random_state.shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]
X_csr = sp.csr_matrix(X)
X_csr.sort_indices()


class MyPerceptron(object):

    def __init__(self, n_iter=1):
        self.n_iter = n_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.n_iter):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))


def plot_data():
    x1 = np.vstack((X[y == 0, 0], X[y == 0, 2]))
    x2 = np.vstack((X[y == 1, 0], X[y == 1, 2]))
    xx = np.vstack((x1.T, x2.T))
    yy = np.array([-1]*50 + [1]*50)
    plt.scatter(xx[yy == -1, 0], xx[yy == -1, 1], c='r')
    plt.scatter(xx[yy == 1, 0], xx[yy == 1, 1], c='b')

    clf = Perceptron(n_iter=100, shuffle=False)
    clf.fit(xx, yy)

    # w0 + w1*x1 + w2*x2 = 0
    # x2 = -w1/w2 * x1 - w0/w1
    xxx = np.arange(xx[:, 0].min() - 0.2, xx[:, 0].max() + 0.2, 0.1)
    yyy = -clf.coef_.ravel()[0]/clf.coef_.ravel()[1]*xxx - clf.intercept_/clf.coef_.ravel()[1]
    plt.plot(xxx, yyy)

    plt.grid()
    plt.show()



def test_perceptron_accuracy():
    for data in (X, X_csr):
        clf = Perceptron(n_iter=30, shuffle=False)
        clf.fit(data, y)
        score = clf.score(data, y)
        print score
        assert_true(score >= 0.7)


def test_perceptron_correctness():
    y_bin = y.copy()
    y_bin[y != 1] = -1

    clf1 = MyPerceptron(n_iter=2)
    clf1.fit(X, y_bin)

    clf2 = Perceptron(n_iter=2, shuffle=False)
    clf2.fit(X, y_bin)

    assert_array_almost_equal(clf1.w, clf2.coef_.ravel())


def test_undefined_methods():
    clf = Perceptron()
    for meth in ("predict_proba", "predict_log_proba"):
        assert_raises(AttributeError, lambda x: getattr(clf, x), meth)


if __name__ == '__main__':

    plot_data()

    #print y
    #test_perceptron_accuracy()
