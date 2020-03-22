# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:33:41 2019

@author: 91953
"""

#%% importing libraries
"""
ALl the necessary packages are called
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
import xgboost as xgb
#from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

#%% Functions Defined

"""
To see how the actual is varying with the predicted across the time
Although the Date column is removed for the easy flow and run of the code
"""
def plotting(key, predictions, expected, mse):
    plt.figure()
    plt.title(key)

    y_values = list(expected) + list(predictions)
    y_range = max(y_values) - min(y_values)
    plt.text(6, min(y_values) + 0.2 * y_range, 'MSE = ' + str(mse))
    plt.plot(predictions)
    plt.plot(expected)
    plt.legend(['predicted', 'expected'])
    plt.show()


#%% Inputing the Datasets

datapath = "C:\\Users\\91953\\Desktop\\Analytic Labs\\Teacheron\\Debodeep\\Stock Market\\Data\\DF1\\"

df1 = pd.read_csv(datapath + "AAPL1.csv")
df2 = pd.read_csv(datapath + "AAPL2.csv")
df3 = pd.read_csv(datapath + "AAPL3.csv")
df4 = pd.read_csv(datapath + "AAPL4.csv")

#%% Printing the Columns of Different Data Frames

print("Dataframe 1 Columns......")
print(list(df1.columns.values))
print("Dataframe 1 Columns......")
print(list(df2.columns.values))
print("Dataframe 1 Columns......")
print(list(df3.columns.values))
print("Dataframe 1 Columns......")
print(list(df4.columns.values))
print("End........")

#%% Removing all the unnecessary columns

df1.pop("Adj Close")
df1.pop("Volume")
df1.pop("Label")

print(list(df1.columns.values))

df2.pop('Open')
df2.pop('High')
df2.pop('Low')
df2.pop("Close")
df2.pop('Adj Close')
df2.pop('Volume')
df2.pop('Label')

print(list(df2.columns.values))

df3.pop('Open')
df3.pop('High')
df3.pop('Low')
df3.pop("Close")
df3.pop('Adj Close')
df3.pop('Volume')
df3.pop('Label')

print(list(df3.columns.values))

df4.pop('Open')
df4.pop('High')
df4.pop('Low')
df4.pop("Close")
df4.pop('Adj Close')
df4.pop('Volume')
df4.pop('Label')

print(list(df4.columns.values))

#%% Performing the Merging of the Datasets to form Universal Dataset

df_merged = pd.concat([df1,df2,df3, df4], axis=1, sort=False)

print(list(df_merged.columns.values))

#%% Bringing in the Test and Training part of the ALgorithms

a,b = df1.shape

train_size = int(a*0.8)
test_size = a-train_size

y = df1.pop('Close')


#%% LSTM for the stock market predictions

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#creating train and test sets
dataset = y.values
inputs = []
train = dataset[0:train_size]
valid = dataset[train_size:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
