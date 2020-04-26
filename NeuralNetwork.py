# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:00:26 2020

@author: Shreya Date
"""

import struct as st
import numpy as np
import ActivationFunctions as af

class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes   = input_nodes
        self.hidden_nodes  = hidden_nodes
        self.output_nodes  = output_nodes
        self.weights_ih    = np.zeros((hidden_nodes, input_nodes)) 
        self.weights_ho    = np.zeros((output_nodes, hidden_nodes))
        self.learning_rate = 0.1 
        
    def feedForward(self, input):
        hidden = np.dot(self.weights_ih, input)     # get wx
        hidden = af.Sigmoid(hidden)                 # activation function
        output = np.dot(self.weights_ho, hidden)
        output = af.Sigmoid(output)
        return input, hidden, output
    
    def train(self, x, y):
        inputs, hidden, yPred = self.feedForward(x)
        
        # calculate output errors
        output_errors = np.subtract(y, yPred)
        
        # calculate gradient
        gradients = np.vectorize(af.Sigmoid_derivative)(yPred)
        print("sigmoid_derivative: ", gradients)
        gradients    = np.dot(gradients, output_errors)
        gradients    = np.multiply(gradients, self.learning_rate)
        
        print("old weights_ih: ", self.weights_ih)
        print("old weights_ho: ", self.weights_ho)

        # calculate deltas
        hidden_t  = np.transpose(hidden)
        weights_ho_delta = np.multiply(gradients, hidden_t)
        self.weights_ho = np.add(self.weights_ho, weights_ho_delta)
        
        # Calculate hidden layer errors
        weights_ho_t  = np.transpose(self.weights_ho)
        hidden_errors = np.dot(weights_ho_t, output_errors)
        hidden_gradients = np.vectorize(af.Sigmoid_derivative)(hidden_t)
        
        # Calculate hidden gradient
        hidden_gradients = np.dot(hidden_gradients, hidden_errors)
        hidden_gradients    = np.multiply(hidden_gradients, self.learning_rate)
        
        # Calculate input->hidden deltas
        inputs_t = np.transpose(inputs)
        weights_ih_delta = np.multiply(hidden_gradients, inputs_t)
        self.weights_ih = np.add(self.weights_ih, weights_ih_delta)
        
        print("updated weights_ih: ", self.weights_ih)
        print("updated weights_ho: ", self.weights_ho)
        
        print(self.weights_ih.shape)
        print(self.weights_ho.shape)
        