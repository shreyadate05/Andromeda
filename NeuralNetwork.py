# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:03:48 2020

@author: Shreya Date
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:00:26 2020

@author: Shreya Date
"""

import numpy
import struct as st
import numpy as np
import scipy.special
import ActivationFunctions as af

class NeuralNetwork:
       
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.weights_ih    = np.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))
        self.weights_ho    = np.random.normal(0.0, pow(self.hnodes, -0.5),(self.onodes, self.hnodes))

        self.lr = learningrate
        self.activation_function = lambda x: af.Sigmoid(x)
        
        self.bias_h = np.ones((self.hnodes, 1))
        self.bias_o = np.ones((self.onodes, 1))


    def train(self, inputs, targets):    
        inputs = numpy.array(inputs, ndmin=2).T
        targets = numpy.array(targets, ndmin=2).T
           
        hidden_inputs    = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden_outputs   = self.activation_function(hidden_inputs)
        
        final_inputs     = np.dot(self.weights_ho, hidden_outputs) + self.bias_o
        final_outputs    = self.activation_function(final_inputs)
        
        output_errors    = targets - final_outputs
        hidden_errors    = np.dot(self.weights_ho.T, output_errors)
        
        output_gradients = output_errors * final_outputs * (1.0 - final_outputs)        
        self.weights_ho += self.lr * np.dot(output_gradients, np.transpose(hidden_outputs))
        self.bias_o      =  self.bias_o + output_gradients
        
        hidden_gradients = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        self.weights_ih += self.lr * np.dot(hidden_gradients, np.transpose(inputs))
        self.bias_h      =  self.bias_h + hidden_gradients
        
        
    def predict(self, inputs):
        hidden_inputs  = np.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs   = np.dot(self.weights_ho, hidden_outputs)
        final_outputs  = self.activation_function(final_inputs)

        return final_outputs
    
   