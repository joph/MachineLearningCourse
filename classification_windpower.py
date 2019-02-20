# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:58:37 2019

@author: jschmidt
"""
import pandas
import os
import numpy as np

from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


os.chdir("G:/Meine Ablage/LVA/PhD Lectures/MachineLearningCourse/") 
df = pandas.read_csv("site-characteristics/ESSO_1km_dk.csv", sep=";")

cap_cells=df['sum_capacity'] > 0

#### proportion of cells with windpower
len(df[cap_cells])/len(df)


sel_cols = [4,7,10,11,12,13,14,15,16,17,18,19,20,21,24,27]


input=(df.iloc[:,sel_cols].values)
labels=(df.iloc[:,3]).values

MinMaxScaler(feature_range=(0, 1))

scaler = MinMaxScaler(feature_range=(0, 1))
#input = scaler.fit_transform(input)

indices = np.random.choice(input.shape[0], round(input.shape[0]*0.8), replace=False)

input_partial=input[indices]
labels_partial=labels[indices]

mask = np.ones(input.shape[0],dtype=bool) #np.ones_like(a,dtype=bool)
mask[indices] = False

input_val=input[mask]
labels_val=labels[mask]

batch_size=1

#input_partial.shape=(round(len(input_partial)/batch_size),batch_size,input_partial.shape[1])

model=Sequential()
model.add(layers.Dense(5,
                       activation='relu',
                       input_shape=(16,)))

model.add(layers.Dense(1))

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit(input_partial,
          labels_partial,
          epochs=10,
          batch_size=batch_size,
          validation_data=(input_val,labels_val))

history_dict = history.history 

loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label="Loss Training")
plt.plot(epochs,val_loss_values,'b',label="Loss Validation")
plt.legend()
plt.show()

plt.scatter(input_partial[:,0],labels_partial)




