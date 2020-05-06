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
    nn = NeuralNetwork(2, 4, 1, 0.1)

    for j in range (100000):
        nn.train([0,0], [0])
        nn.train([0,1], [1])
        nn.train([1,0], [1])
        nn.train([1,1], [0])
             
    o1 = nn.predict([0,0])
    o2 = nn.predict([0,1])
    o3 = nn.predict([1,0])
    o4 = nn.predict([1,1])
     
    print(o1)
    print(o2)
    print(o3)
    print(o4)
 
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

