# -*- coding:utf-8 -*-

import numpy as np


class Sigmoid(object):
    '''
    Sigmoid 激活函数
    '''

    def forward(self, input):
        return 1.0 / (1.0 + np.exp(-input))

    def backward(self, output):
        return self.forward(output) * (1 - self.forward(output))