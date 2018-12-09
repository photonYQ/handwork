# -*- coding:utf-8 -*-
import gzip
import cPickle
import numpy as np


def data_trans(n):
    tmp = [0] * 10
    tmp[n] = 1
    return tmp

def data_loader():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    train_data, valid_data, test_data = cPickle.load(f)
    data = []
    for dataset in [train_data, valid_data, test_data]:
        data.append([dataset[0].T, np.array([data_trans(i) for i in dataset[1]]).T])
    return data
    

if __name__ == '__main__':
    tr, v, t = data_loader()
    print t[1][:,1]


