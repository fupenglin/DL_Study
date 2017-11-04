# -*- coding:utf-8 -*-

import numpy as np


class Perceptron(object):

    def __init__(self, input_num, activator):
        self.activator = activator
        self.w = 2 * np.random.random(input_num) - 1
        self.b = 0

    def predict(self, input_data):
        # 预测值
        a = np.dot(input_data, self.w.T) + self.b
        for i in range(len(a)):
            a[i] = self.activator(a[i])
        return a.T

    def train(self, input_data, labels, iteration, rate):
        # 根据输入数据训练感知器
        for _ in range(iteration):
            self._one_iteration(input_data, labels, rate)

    def _one_iteration(self, input_data, labels, rate):
        # 一次迭代计算训练，并更新权值
        output_data = self.predict(input_data)
        self._update_weights(input_data, output_data, labels, rate)

    def _update_weights(self, input_data, output_data, labels, rate):
        delta = labels - output_data

        # 更新权值
        d_w = delta.dot(input_data) / input_data.shape[0]
        self.w += d_w * rate

        # 更新偏置值
        d_b = delta.sum(axis=0) / input_data.shape[0]
        self.b += d_b * rate


def f(a):
    return 1 if a >= 0 else 0


def get_train_data():
    input_data = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([1, 0, 0, 0])
    return input_data, labels


def train_and_perceptron():
    input_data, labels = get_train_data()
    and_p = Perceptron(input_data.shape[1], f)
    and_p.train(input_data, labels, 1000, 0.11)
    return and_p


if __name__ == '__main__':
    p = train_and_perceptron()
    print "1 and 1 = ", p.predict([[1, 1]])
    print "1 and 0 = ", p.predict([[1, 0]])
    print "0 and 1 = ", p.predict([[0, 1]])
    print "0 and 0 = ", p.predict([[0, 0]])
