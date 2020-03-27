import numpy as np
import pandas as pd
import math

class Neural_Net:
    def __init__(self,hidden_layers,layer_size,activation):
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.activation = activation
        self.weights = []
        self.io_shape = (10,2)

    def build_network(self):

        for layer in range(self.hidden_layers):
            if layer == 0:
                size = (self.io_shape[0],self.layer_size+1)
            elif layer == self.hidden_layers-1:
                 size = (self.layer_size+1,self.io_shape[0])
            else:
                size = (self.layer_size+1,self.layer_size+1)
            
            self.weights.append(np.random.standard_normal(size=size))
            self.weights[layer][:,-1]=1
    
    def forward_pass(self,x):
        for W in weights:
            x*=W
        return x

    def train(self,x_train,y_true):
        return 'trained weights'

    def test(self,x_test,y_true):
        return 'accuracy'

    def predict(self,x):
        return 'prediction'