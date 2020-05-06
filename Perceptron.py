# =============================================================================
# Shreya Date
# CMSC-678: Introduction to Machine Learning
# Homework 2
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTS:

import numpy as np
from sklearn.model_selection import train_test_split
import struct as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt 
from sklearn.metrics import average_precision_score

# -----------------------------------------------------------------------------
# GLOBAL VARIABLES

image_size = 28
no_of_different_labels = 10
image_pixels = image_size * image_size
X = y = x_train = x_test = y_train = y_test = w = wx = np.empty(0)
y_predicted = accuracy = 0
training_accuracy = test_accuracy = []
epoch = 1


xTrain = yTrain = xTest = yTest = np.empty(0)
xTrain_file = 'data/train-images.idx3-ubyte'
xTest_file  = 'data/t10k-images.idx3-ubyte'
yTrain_file = 'data/train-labels.idx1-ubyte'
yTest_file  = 'data/t10k-labels.idx1-ubyte'

# -----------------------------------------------------------------------------
# APIs

def img(row, data):
	image = np.zeros((image_size,image_size))
	for i in range(0,image_size):
		for j in range(0,image_size):
			pix = image_size*i+j
			image[i,j] = data[row, pix]
	plt.imshow(image, cmap = 'gray')
	plt.show()
            
def calculate_accuracy(y):
    global accuracy
    return (accuracy/y.shape[0])
    
def calculate_wx(x):
    global w, wx
    for i in range(len(w)):
        wx[i] = (np.dot(x, w[i]))
    #print(wx.shape)
    #print(w.shape)
    #print(x.shape)
    return np.argmax(wx)

    
def test_perceptron(x, y):
    global w, wx, y_predicted, accuracy
    scorecard = []
    actual = []
    for i in range(len(x)):
        y_predicted = calculate_wx(x[i])
        y_actual    = np.argmax(y[i])
        actual.append(y_actual)
        if (y_predicted == y_actual):
            accuracy = accuracy + 1
            scorecard.append(1)
        else:
            scorecard.append(0)
    print("Accuracy is: ", calculate_accuracy(y))
    average_precision = average_precision_score(scorecard,  actual)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    accuracy = 0

def perceptron():
    global w, x_train, y_train, wx, y_predicted
    print(len(x_train[1]))
    print(x_train[1].shape)
    for i in range(len(x_train)):
        y_predicted = calculate_wx(x_train[i])
        y_actual    = np.argmax(y_train[i])
        if (y_predicted != y_actual):
            w[y_predicted] = w[y_predicted] - x_train[i] 
            w[y_actual]    = w[y_actual]    + x_train[i]

def init():
    global w, x_train, wx, no_of_different_labels, x_test
    x_train = np.append(x_train, np.ones([len(x_train),1]),1)
    x_test  = np.append(x_test, np.ones([len(x_test),1]),1)
    w = np.zeros((no_of_different_labels, x_train.shape[1]))
    wx =  np.zeros((no_of_different_labels, 1))
    print(x_train.shape)
        

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


def load_mnist():
    global xTrain_file, xTest_file, yTrain_file, yTest_file
    global x_train, y_train, x_test, y_test 
    
    x_train = getData(xTrain_file)
    x_train = x_train.flatten().reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    
    x_test  = getData(xTest_file)
    x_test  = x_test.flatten().reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
        
    y_train = getData(yTrain_file)
    y_test  = getData(yTest_file)
    
    y_train = oneHotY(y_train)
    y_test  = oneHotY(y_test)

   
# -----------------------------------------------------------------------------
# DRIVER
    
def main():
    global y_test, y_train
    load_mnist()
    init()
    #print(y_test)
    for i in range(epoch):
        perceptron()
    #test_perceptron(x_train, y_train)
    test_perceptron(x_test, y_test)

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------