# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:40:37 2020

@author: Shreya Date
"""
import numpy as np
from NeuralNetwork import NeuralNetwork


xTrain = yTrain = xTest = yTest = np.empty(0)

def setup():  
    global xTrain, yTrain, xTest, yTest 
    nn = NeuralNetwork(2, 2, 1)
    print("data: ", xTrain, xTrain.shape)
    i = 0
    for data in xTrain:
        print("data: ", data, data.shape)
        print(yTrain[i])
        nn.train(xTrain[i], yTrain[i]) 
        i = i + 1

def init():
    global xTrain, yTrain, xTest, yTest 
    
    xTrain = np.array([[0,0], [1,0], [0,1], [1,1]])
    xTest = np.array([[0,0], [1,0], [0,1], [1,1]])
        
    yTrain = np.array([0, 1, 1, 0])
    yTest  = np.array([0, 1, 1, 0])
    
def main():
    init()
    setup()
    
if __name__ == "__main__":
    main()

