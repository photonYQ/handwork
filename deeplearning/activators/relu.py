# -*- coding:utf-8 -*-
import numpy as np


class ReLU(object):
    '''
    ReLU 激活函数
    '''

    def forward(self, input):
        return max(0, input)

    def backward(self, output):
        return 1 if output > 0 else 0


