# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:15:16 2019

@author: joph
"""

import numpy as np
import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
#from pandas_datareader import data
import json
import urllib.request
import math
import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import LSTM
from keras.optimizers import adam

from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pydot

        

class Function_data_no_ts:
    
    def __init__(self,n,upper,max_elem,model,f,lags):
        
        self.model=model
        self.upper=upper
        self.max_elem=max_elem
        self.f=f
        self.lags=lags
        x=np.random.random(n)*upper
        x=np.sort(x)
        y=self.function(x)
        self.dataset= [*zip(*[x,y])]
        
        
        lagged=np.zeros((len(x)-self.lags,self.lags))

        for i in range(0,len(x)-lags):
            lagged[i,:]=x[i:i+lags]
            
        y=y[0:(len(x)-lags)]
        y.shape=(len(x)-lags,1)
        self.dataset = np.append(lagged,y,1)
        
        print(len(self.dataset))
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset_scaled = self.scaler.fit_transform(self.dataset)
        
        self.dataset_scaled_short = self.dataset_scaled[0:max_elem,:]
        self.dataset_out_of_range = self.dataset_scaled[max_elem:self.dataset_scaled.shape[0],:]
        
        self.train, self.test = train_test_split(self.dataset_scaled_short,train_size=0.7,test_size=0.3)
        print("test")
        #print(self.train.shape)
        #print(self.test.shape)
        
    
    def function(self,x):
        return self.f(x)
    
    def train_data(self):
          return self.train
      
    def test_data(self):
        return self.test
    
    def fit_model_plot_results(self,epochs,batch_size):
         x=self.train[:,0:(self.train.shape[1]-1)]
         y=self.train[:,self.train.shape[1]-1]
         
         self.train_model(x,y,epochs,batch_size)

    def train_model(self,x,y,epochs,batch_size):
        
         index=np.random.randint(0,len(x),round(0.8*len(x)))
        
         x_t=x[index]
         y_t=y[index]
         
         mask = np.ones(len(x),dtype=bool) #np.ones_like(a,dtype=bool)
         mask[index] = False
         
         x_val=x[mask]
         y_val=y[mask]
         
        
         self.history=self.model.fit(x,
                                     y,
                                     epochs=epochs,
                                     batch_size=batch_size,
                                     validation_data=(x_val,y_val))
         
         
         loss=self.history.history['loss']
         val_loss=self.history.history['val_loss']
         epochs=range(1,len(loss)+1)
         plt.plot(epochs,loss,'bo')
         plt.plot(epochs,val_loss,'b')

         plt.title('Model loss')
         plt.ylabel('Loss')
         plt.xlabel('Epoch')
         plt.legend(['Train', 'Test'], loc='upper left')
         plt.show()

         self.plot_test_results(batch_size)
         self.plot_out_of_range_results(batch_size)
         
   
    def plot_test_results(self,batch_size):
         x=self.test[:,0:(self.test.shape[1]-1)]
         self.plot_test_results_int(x,batch_size)
         
    def plot_test_results_int(self,x,batch_size):
       
         y_predict_test=self.model.predict(x,batch_size=batch_size)
         y_predict_test.shape = (len(y_predict_test))
         
         dataset_invert = np.copy(self.test)
         dataset_invert[:,dataset_invert.shape[1]-1]=y_predict_test
         y_predict_test= self.scaler.inverse_transform(dataset_invert)[:,dataset_invert.shape[1]-1]
         
         test_tf=self.scaler.inverse_transform(self.test)
         x=test_tf[:,0]
         y=test_tf[:,test_tf.shape[1]-1]
         
         plt.scatter(x,y)
         plt.scatter(x,y_predict_test)
         plt.show()

    def plot_out_of_range_results(self,batch_size):
        
         x=self.dataset_out_of_range[:,0:(self.dataset_out_of_range.shape[1]-1)]
         self.plot_out_of_range_results_int(x,batch_size)
         
    def plot_out_of_range_results_int(self,x,batch_size):
         
         y_predict_out_range=self.model.predict(x,batch_size=batch_size)
         y_predict_out_range.shape = (len(y_predict_out_range))
         
         dataset_invert = np.copy(self.dataset_out_of_range)
         dataset_invert[:,dataset_invert.shape[1]-1]=y_predict_out_range
         y_predict_out_range= self.scaler.inverse_transform(dataset_invert)[:,dataset_invert.shape[1]-1]
         
         test_tf=self.scaler.inverse_transform(self.dataset_out_of_range)
         x=test_tf[:,0]
         y=test_tf[:,test_tf.shape[1]-1]
         
         plt.scatter(x,y)
         plt.scatter(x,y_predict_out_range)
         plt.show()
        
        

class Function_data_lstm(Function_data_no_ts):
    
    def __init__(self,n,upper,max_elem,model,f,lags):
        super().__init__(n,upper,max_elem,model,f,lags)
    
    def fit_model_plot_results(self,epochs,batch_size):
         x=self.train[:,0:(self.train.shape[1]-1)]
         y=self.train[:,self.train.shape[1]-1]
         
         x=x.reshape((self.train.shape[0], self.lags, 1))
         
         self.train_model(x,y,epochs,batch_size)
         
    def plot_test_results(self,batch_size):
         x=self.test[:,0:(self.test.shape[1]-1)]
         x=x.reshape((self.test.shape[0], self.lags, 1))
  
         self.plot_test_results_int(x,batch_size)
         
    def plot_out_of_range_results(self,batch_size):
        
         x=self.dataset_out_of_range[:,0:(self.dataset_out_of_range.shape[1]-1)]
         x=x.reshape((self.dataset_out_of_range.shape[0], self.lags, 1))
         self.plot_out_of_range_results_int(x,batch_size)
   
        
        
        
        
        

    
    
    

def train_x_y_model(dataset,model,lstm=0,epochs=10,batch_size=1):

    #model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae','acc'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    train_size = round(len(dataset)*0.7)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
    
    x=train[:,0:(train.shape[1]-1)]
    y=train[:,train.shape[1]-1]
    x_test=test[:,0:(test.shape[1]-1)]
    y_test=test[:,test.shape[1]-1]
    x.shape=(train_size,train.shape[1]-1)
    y.shape=(train_size,1)
    x_test.shape=(test_size,test.shape[1]-1)
    y_test.shape=(test_size,1)
    x1=0
    x_test1=0
    if(lstm==1):
        x1=x
        x_test1=x_test
        x = x[0:train_size].reshape((train_size, 1, 1))
        x_test=x_test[0:test_size].reshape((test_size, 1, 1))
        
   
    dataset_after = scaler.inverse_transform(dataset)

    y_predict_train= model.predict(x,batch_size=batch_size)
    y_predict_test= model.predict(x_test,batch_size=batch_size)

    y_predict_train.shape = (len(y_predict_train))
    dataset1 = np.copy(dataset_after[0:train_size,:])
    dataset1[:,dataset1.shape[1]-1]=y_predict_train
    y_predict_train= scaler.inverse_transform(dataset1)[:,dataset1.shape[1]-1]
    
    train, test = dataset_after[0:train_size,:], dataset_after[train_size:len(dataset),:]
    print(len(train), len(test))
    
    x=train[:,0:(train.shape[1]-1)]
    y=train[:,train.shape[1]-1]
    x_test=test[:,0:(test.shape[1]-1)]
    y_test=test[:,test.shape[1]-1]
    x.shape=(train_size,train.shape[1]-1)
    y.shape=(train_size,1)
    x_test.shape=(test_size,test.shape[1]-1)
    y_test.shape=(test_size,1)
    x1=0
    x_test1=0
    if(lstm==1):
        x1=x
        x_test1=x_test
        x = x[0:train_size].reshape((train_size, 1, 1))
        x_test=x_test[0:test_size].reshape((test_size, 1, 1))
        
    
    if(lstm==1):
        plt.scatter(x1,y)
        plt.scatter(x1,y_predict_train)
    else:
        plt.scatter(x,y)
        plt.scatter(x,y_predict_train)
   
    plt.show()
    
    y_predict_test.shape = (len(y_predict_test))
    dataset1 = np.copy(dataset[0:test_size,:])
    dataset1[:,dataset1.shape[1]-1]=y_predict_test
    y_predict_test= scaler.inverse_transform(dataset1)[:,dataset1.shape[1]-1]
    
    
    if(lstm==1):
        plt.scatter(x_test1,y_test)
        plt.scatter(x_test1,y_predict_test)
    else:
        
        plt.scatter(x_test,y_test)
        plt.scatter(x_test,y_predict_test)
   
    plt.show()
   
    #plt.plot(m.history['acc'])
    #plt.plot(m.history['val_acc'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper left')
    #plt.show()

    # Plot training & validation loss values
    plt.plot(m.history['loss'])
    #plt.plot(m.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return model