# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:00:26 2020

@author: Shreya Date
"""

import struct as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork import NeuralNetwork


xTrain = yTrain = xTest = yTest = np.empty(0)
xTrain_file = 'data/train-images.idx3-ubyte'
xTest_file  = 'data/t10k-images.idx3-ubyte'
yTrain_file = 'data/train-labels.idx1-ubyte'
yTest_file  = 'data/t10k-labels.idx1-ubyte'

def setup():  
    nn = NeuralNetwork(xTrain.shape[1], 512, 10)
    nn.train(xTrain[1], yTrain[1]) 
    
def getData(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
def oneHotY(y):
    label_encoder   = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder  = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded  = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def init():
    global xTrain_file, xTest_file, yTrain_file, yTest_file
    global xTrain, yTrain, xTest, yTest 
    
    xTrain = getData(xTrain_file)
    xTrain = xTrain.flatten().reshape(xTrain.shape[0], xTrain.shape[1]*xTrain.shape[2])
    xTrain = np.append(xTrain, np.ones([len(xTrain),1]),1)
    
    xTest  = getData(xTest_file)
    xTest  = xTest.flatten().reshape(xTest.shape[0], xTest.shape[1]*xTest.shape[2])
    xTest  = np.append(xTest, np.ones([len(xTest),1]),1)
        
    yTrain = getData(yTrain_file)
    yTest  = getData(yTest_file)
    
    yTrain = oneHotY(yTrain)
    yTest  = oneHotY(yTest)
    
    
def main():
    init()
    setup()
    
if __name__ == "__main__":
    main()

