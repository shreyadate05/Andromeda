# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:59:15 2020

@author: Shreya Date
"""

import numpy as np


def Sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def Sigmoid_derivative(x):
    s = Sigmoid(x)
    ds = s*(1-s)
    return ds

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(values):
    result = [1 if x > 0 else 0 for x in values]
    return result

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - (np.tanh(x) ** 2)

def Softmax(x):
    return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))

def Softmax_derivative(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
