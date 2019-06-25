# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:05:51 2019

@author: Mohit
"""
# =============================================================================
#  Steps:
#      1. Extract data
#      2. Data Cleaning
#      3. identify X and Y
#      4. split into train and test
#      5. define deep learning model
#      6. fit your model with train and test data
#      7. predict for test data
#      8. find accuracy of model
# =============================================================================


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:/Users/Mohit/Desktop/Nikhil Analytics Material/Python/Deep Learning/Day 1/Bank_Churn_Modelling.csv")
data.head()

data.columns
# to convert character value to numeric values - 
# labelencoding method - in this method we transform character values to 0,1,2,3...
from sklearn.preprocessing import LabelEncoder
le_Gender=LabelEncoder()
le_Gender.fit(data['Gender'].unique())
data['Gender']=le_Gender.transform(data['Gender'])

le_Geography=LabelEncoder()
le_Geography.fit(data['Geography'].unique())
data['Geography']=le_Geography.transform(data['Geography'])

data.isnull().sum()

X = data.iloc[:,3:13]
Y = data.iloc[:,13]

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=10)

model=keras.Sequential([
    layers.Dense(64, activation=tf.nn.sigmoid, input_shape=[len(train_x.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(2,activation=tf.nn.sigmoid),
  ])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x,train_y,epochs=5)

pred_y=model.predict(test_x)
y_pred=[]
for i in range(len(pred_y)):
    y_pred.append(np.argmax(pred_y[i]))
    
accuracy_score(test_y,y_pred)

