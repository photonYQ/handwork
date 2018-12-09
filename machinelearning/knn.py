# -*- coding:utf-8 -*-
# MNIST问题KNN算法解
from __future__ import division
import numpy as np
from data_loader import data_loader


class KnnClassifier(object):

    def __init__(self, p=2, k=1):
        '''
        默认取L2距离，最近邻解
        '''
        self.p = p
        self.k = k

    def set_train(self, train_data, train_label):
        '''
        设置训练数据集
        :param train_data: 训练数据输入 nxm n维特征空间，m个样本
        :param train_label: 训练数据实际分类 1xm m个样本
        '''
        self.train_data = train_data
        self.train_label = train_label

    def predict(self, x):
        '''
        预测x对应的分类
        :param x: 待预测的输入 nx1 n维特征空间
        :return: 训练集中最接近测试样本的数据下标
        '''
        kmax = np.argmin(np.sum((self.train_data - x)**2, axis=0))
        return kmax

    def evaluate(self, test_data, test_label):
        '''
        评估测试数据集结果
        :param test_data: 测试数据输入 nxm n维特征空间，m个样本
        :param test_label: 测试数据实际分类 1xm m个样本
        :return: 测试数据集预测准确率
        '''
        num_scale, num_test = test_data.shape
        score = 0
        for i in xrange(num_test):
            predict = self.predict(test_data[:,i].reshape(num_scale,1))
            if np.argmax(test_label[:,i]) == np.argmax(self.train_label[:,predict]):
                print "%d/%d" % (score, i+1)
                score += 1
        return score / num_test


if __name__ == '__main__':
    train_data, validate_data, test_data = data_loader()
    knn = KnnClassifier(2, 1)
    knn.set_train(train_data[0], train_data[1])
    result = knn.evaluate(test_data[0], test_data[1])
    print result
    
