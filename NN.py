import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

class Neural_Net:
    def __init__(self,hidden_layers,layer_size,shape):
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.weights = []
        self.activations=[]
        self.io_shape = shape
        self.build_network()
        self.lossMA = []
        
    # Activations
    
        
    def build_network(self):

        for layer in range(self.hidden_layers):
            if layer == 0:
                size = (self.io_shape[0],self.layer_size)
            elif layer == self.hidden_layers-1:
                 size = (self.layer_size,self.io_shape[1])
            else:
                size = (self.layer_size,self.layer_size)
            
            self.weights.append(np.random.standard_normal(size=size))
            self.activations.append(np.zeros(shape=(1,1)))
            self.weights[layer][:,-1]=1
            
    
    """ Forward pass, also used for prediction"""
    def forward_pass(self,x):
        for W in range(len(self.weights)):
            if W != (len(self.weights)-1):#relu
                x = abs(np.matmul(x,self.weights[W]))
            else:#sigmoid
                x = np.matmul(x,self.weights[W])
                x = np.clip(x, -10,10)
                x = (1.0/(1.0+np.exp(-1.0*x)))
            self.activations[W]=x
        return x

    """ The log loss function"""
    def log_loss(self,y_true, y_pred, eps = 1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = 0
        #assert(y_true==y_pred)
        for k in range(len(y_true)):
          if y_true[k] == 1:
            loss+=(-np.log(y_pred[k]))
          else:
            loss+=(-np.log(1 - y_pred[k]))
        return loss[0]
    
    #very simple back prop
    def backpropagation(self,alpha,beta,loss):
        self.lossMA.append(loss)
        MA = np.mean(self.lossMA)
        for W in range(len(self.weights)):
            self.weights[W] = self.weights[W] - (alpha*loss+(beta*MA))
        return loss
        
    def train(self,x_train,y_true):
        return 'trained weights'

    def test(self,x_test,y_true):
        return 'accuracy'

    def predict(self,x):
        return self.forward_pass(x)


def Main():
    #init data
    #x = np.random.normal(size = (10,10))
    #y = np.array([0,1,1,0,1,1,1,0,1,0],dtype='float16').reshape((-1,1))
    data = pd.read_csv("train.csv")
    y = data.label
    del data["label"]
    x_train,x_test,y_train,y_test = train_test_split(data,y,
                                                     test_size=0.20,random_state=0)
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train =np.array(y_train).reshape((-1,1))
    y_test =np.array(y_test).reshape((-1,1))
    #Init neural net    
    del data
    del y
    NN1 = Neural_Net(hidden_layers=6,layer_size=x_train.shape[1],shape=(x_train.shape[1],y_train.shape[1]))
    
    print(NN1.weights[0].shape)
    print(NN1.io_shape)
    
    for k in range(200):
        ypred = NN1.predict(x_train)
        loss = NN1.log_loss(y_train,ypred)
        NN1.backpropagation(0.02,0.98,loss)
        print("Epoch: %s, loss: %s"%(k,loss))
    print("--------------------------")
    
    y_test_pred = NN1.predict(x_test)
    print("Test Loss: %s"%NN1.log_loss(y_test,y_test_pred))
    return ypredy_test_pred 

pred Main()

