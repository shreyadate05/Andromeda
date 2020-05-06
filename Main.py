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
from sklearn.metrics import average_precision_score
from sklearn import metrics

xTrain = yTrain = xTest = yTest = np.empty(0)
xTrain_file = 'data/train-images.idx3-ubyte'
xTest_file  = 'data/t10k-images.idx3-ubyte'
yTrain_file = 'data/train-labels.idx1-ubyte'
yTest_file  = 'data/t10k-labels.idx1-ubyte'
epoch       = 1
nn          = NeuralNetwork(784, 200, 10, 0.1)   

yA = []
yP = []

def test(xTrue, yTrue):
    global nn, yA, yP
    j = 0
    scorecard = []
    for image in xTrue:
        inputs = (np.asfarray(image) / 255.0 * 0.99) + 0.01
        y = nn.predict(inputs)
        yPred = np.argmax(y)
        yReal = np.argmax(yTrue[j])
        yA.append(yReal)
        yP.append(yPred)
        if(yPred == yReal):
            scorecard.append(1)
        else:
            scorecard.append(0)
        j += 1
    return scorecard


def calculate_training_accuracy():
    global xTrain, yTrain, xTest, yTest, accuracy, nn
    scorecard = test(xTrain, yTrain)
    accuracy = (sum(scorecard)/len(scorecard))
    print("Correct Predictions: ", sum(scorecard))        
    print("Total   Predictions: ", len(scorecard))        
    print("Training Accuracy is: ", accuracy*100, "%")  


def calculate_test_accuracy():
    global xTrain, yTrain, xTest, yTest, accuracy, nn, yA, yP
    scorecard = test(xTest, yTest)
    accuracy = (sum(scorecard)/len(scorecard))
    print("Correct Predictions: ", sum(scorecard))        
    print("Total   Predictions: ", len(scorecard))        
    print("Test Accuracy is: ", accuracy*100, "%")
    print(metrics.classification_report(yA, yP, digits=3))
    print(metrics.classification_report(yA, yP, digits=3))
    
    
def train():  
    global xTrain, yTrain, xTest, yTest, epoch, nn
    nn = NeuralNetwork(784, 200, 10, 0.1)        
    for i in range(epoch):
        j = 0
        for image in xTrain:
            inputs = (np.asfarray(image) / 255.0 * 0.99) + 0.01
            nn.train(inputs, yTrain[j])
            j += 1
        print("Completed Epoch ", i)
   
         
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
    
    xTest  = getData(xTest_file)
    xTest  = xTest.flatten().reshape(xTest.shape[0], xTest.shape[1]*xTest.shape[2])
        
    yTrain = getData(yTrain_file)
    yTest  = getData(yTest_file)
    
    yTrain = oneHotY(yTrain)
    yTest  = oneHotY(yTest)

        
def main():
    init()       
    train()
    calculate_training_accuracy()
    calculate_test_accuracy()

    
if __name__ == "__main__":
    main()
