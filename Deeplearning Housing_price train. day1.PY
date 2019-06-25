# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:11:05 2019

@author: DELL
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data=pd.read_csv("H:/Nikhil analytics/PYTHON file/Deep Learning/Day 1/california_housing_train.csv")
print(data)
print(data.shape)
#17000,9
data.isnull().sum()
# no missing data present in data set

X=data.iloc[:,:8]
Y=data.iloc[:,8]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

def norm(x):
    return((x-x.mean())/x.std());
train_x_norm=norm(train_x)
train_y_norm=norm(train_y)

model=keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[8]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_absolute_error', 
                       'mean_squared_error'])

model.fit(train_x_norm,train_y_norm,epochs=6)

test_x_norm=norm(test_x)
test_y_norm=norm(test_y)

pred_y=model.predict(test_x_norm)

from sklearn.metrics import mean_squared_error

mean_squared_error(test_y_norm,pred_y)












