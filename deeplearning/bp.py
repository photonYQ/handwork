# -*- coding:utf-8 -*-
#
# Feedforward neural network based on back propagation
#
# authored by lingkong.dyq at 2018-08-27

# 采用矩阵算法
import random
import numpy as np

from activators.sigmoid import Sigmoid
from data_loader import data_loader


class FeedforwardNeuralNetwork(object):
    '''
    基于反向传播算法的前馈型深度神经网络
    '''

    def __init__(self, dimentions, rate, activator):
        '''
        初始化函数
        :param dimentions: 维度列表，分别为输入(0)层、第1层、...、输出(L)层维度
        :param rate: 学习速率
        :param activator: 神经元
        '''
        self.num_layers = len(dimentions)
        self.dim_layers = dimentions
        self.learn_rate = rate
        self.activator = activator

    def init_params(self):
        '''
        初始化参数矩阵
        :return:
        '''
        # 随机初始化系数矩阵和偏置矩阵
        self.w = [np.random.randn(self.dim_layers[i], self.dim_layers[i+1]) for i in range(self.num_layers - 1)]
        self.b = [np.random.randn(dim, 1) for dim in self.dim_layers[1:]]

    def forward(self, input_data):
        '''
        前向传播，计算每层的z值和a值(activation)
        :param train_data: 训练输入数据 格式为n*m矩阵(numpy.ndarray) n为输入维度 m为样本数
        :return:
        '''
        # 塞入输入层数据
        self.a[0] = input_data
        for i in range(self.num_layers-1):
            self.z[i+1] = np.dot(self.w[i].T, self.a[i]) + self.b[i]
            self.a[i+1] = self.activator.forward(self.z[i+1])

    def backprop(self, output_data):
        '''
        反向传播，计算每层的参数下降梯度
        :param output_data: 训练输出数据 格式为n*m矩阵(numpy.ndarray) n为输出维度 m为样本数
        :return:
        '''
        num_samples = output_data.shape[1]
        # 输出层delta
        delta_z = (self.a[-1] - output_data) / float(num_samples)
        for i in range(self.num_layers-1, 0, -1):
            # 计算第i层系数梯度
            delta_w = np.dot(self.a[i-1], delta_z.T) / float(num_samples)
            delta_b = np.sum(delta_z, axis=1, keepdims=True) / float(num_samples)
            # 更新第i层系数矩阵
            self.w[i-1] = self.w[i-1] - self.learn_rate * delta_w
            self.b[i-1] = self.b[i-1] - self.learn_rate * delta_b
            # 更新delta_z
            delta_z = np.dot(self.w[i-1], delta_z) * self.activator.backward(self.z[i-1])

    def shuffle_data(self, data, seed):
        '''
        乱序训练数据
        :param data: 训练数据
        :param seed: 随机种子
        :return: 乱序后的数据
        '''
        tmp = zip(data[0].T, data[1].T)
        random.Random(seed).shuffle(tmp)
        return [np.array(sample).T for sample in zip(*tmp)]

    def train(self, iterations, train_data, test_data):
        '''
        训练网络
        :param iterations: 迭代次数
        :param test_data: 训练数据 格式 (输入，输出)
                          其中输入输出均为n*m矩阵(numpy.ndarray) n为输入输出维度 m为样本数
        :return:
        '''
        # 每次训练重新初始化参数矩阵
        self.init_params()
        for i in range(iterations):
            # 乱序训练数据
            shuffled_data = self.shuffle_data(train_data, i)
            num_samples = shuffled_data[1].shape[1]
            # 初始化缓存矩阵 缓存每层z值和a值
            self.z = [np.zeros([dim, num_samples]) for dim in self.dim_layers]
            self.a = [np.zeros([dim, num_samples]) for dim in self.dim_layers]
            # 前向传播
            self.forward(shuffled_data[0])
            # 反向传播
            self.backprop(shuffled_data[1])
            print "%dth evaluation: " % i
            self.evaluate(test_data)

    def evaluate(self, test_data):
        '''
        验证已学习的神经网络
        :param test_data: 测试数据 格式 (输入，输出)
                          其中输入输出均为n*m矩阵(numpy.ndarray) n为输入输出维度 m为样本数
        :return:
        '''
        inputs, outputs = test_data
        a = inputs
        for i in range(self.num_layers-1):
            z = np.dot(self.w[i].T, a) + self.b[i]
            a = self.activator.forward(z)
        predict = np.argmax(a, axis=0)
        outputs = np.argmax(outputs, axis=0)
        point = 0
        for i in range(outputs.size):
            if predict[i] == outputs[i]:
                point += 1
        num_samples = len(outputs)
        print "%d/%d" % (point, num_samples)

if __name__ == '__main__':
    train_data, validate_data, test_data = data_loader()
    activator = Sigmoid()
    network = FeedforwardNeuralNetwork([784, 30, 10], 80000, activator)
    network.train(1000, train_data, test_data)
