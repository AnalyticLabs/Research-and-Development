# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:24:13 2019

@author: AKE9KOR
"""

#%%
#Importing libraries
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%
'''
Function used to fit the model



'''
def fit_model(train_data, model):
#    train_data=train_data.values

    X, y = train_data[:, 0:-1], train_data[:, -1]
#    print(X.shape,y.shape)
    model.fit(X, y)

    return model

#%%
'''
Function used for one-step forecasting


'''
def forecast_model(model, X):

    yhat = model.predict(X)

    return yhat

#%%
'''
Function used for plotting

'''
def plotting(key, predictions, expected):
#    pass
    rmse = calculate_rmse(key, expected, predictions)
#    expected.reset_index(drop = True, inplace = True)
    plt.figure()
    plt.title(key)

    y_values = list(expected) + list(predictions)
    y_range = max(y_values) - min(y_values)
    plt.text(6, min(y_values) + 0.2 * y_range, 'RMSE = ' + str(rmse))
    plt.plot(predictions)
    plt.plot(expected)
    plt.legend(['predicted', 'expected'])
    plt.show()

#%%
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = sum(np.abs((y_true - y_pred) / y_true)* 100)/len(y_true)
    if np.isnan(mape)== True or np.isfinite(mape)==False:
        mape=0
    return mape

#%%
def calculate_facc(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    facc = 1 - (sum(np.abs(y_true - y_pred)) / sum(y_true))
    if np.isnan(facc)== True or np.isfinite(facc)==False:
        facc=0
    return facc * 100
#%%


def calculate_rmse(key, expected, predictions):
    expected=np.array(expected)
    rmse = sqrt(mean_squared_error(expected, predictions))
    print("RMSE FOR %s: %d " % (key, rmse))
    return rmse

#%%
def moving_average(test,n,n1):
    train = []
#    test=np.array(test)
#    for num in test:
#        t = num
#        train.append(t)
    train = [x for x in test]
    pred = []
    for num in range(n):
        test_new = pd.DataFrame(train)
        pred1 = (test_new.tail(n1).mean())
        pred1 = pred1[0]
        pred.append(pred1)
        train.append(pred1)

    return pred

#%%
def weighted_moving_average(test1,n,n1):
#    print("Weighted moving average")
    alpha=[0.25,0.3,0.45]
    train = [x for x in test1]
    pred = []
    for num in range(n):
        test_new = pd.DataFrame(train)
        pred1 = test_new.tail(n1)
        pred1=np.dot(pred1[0],alpha)
        pred.append(pred1)
        train.append(pred1)
    pred = [int(i) for i in pred]
    return pred
#%%
def check_repetition(arr, limit, index_start, index_end):
    length = index_start
    try:
        for i in range(0, int( len(arr)/length)):
            condition  = np.array( arr[i:int(i+length)]) - np.array( arr[int( i+length):int(i+2*length)])
            condition  = np.sum([abs(number) for number in condition])
            if condition >= limit :
                if length + 1 <= index_end:
                    return check_repetition(arr, limit, length + 1, index_end)
            # if not than no more computations needed
                else:
                    return 0

            if i == int( len(arr)/length)-2:
                return(length)
    except:
        for i in range(0, int( len(arr)/length)):
            if  i+2*length+1 <= index_end and i+length+1 <= index_end:
                break
            condition  = np.array( arr[i:int(i+length)]) - np.array( arr[int( i+length):int(i+2*length)])
            condition  = np.sum([abs(number) for number in condition])
            if condition >= limit :
                if length + 1 <= index_end:
                    return check_repetition(arr, limit, length + 1, index_end)
            # if not than no more computations needed
                else:
                    return 0

            if i == int( len(arr)/length)-2:
                return(length)

    return 0
#%%
def Croston(dataset,forecast_period=1,alpha=0.4):
    d = np.array(dataset) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d, [np.nan] * forecast_period) # Append np.nan into the demand array to cover future periods

    #level (a), periodicity(p) and forecast (f)
    a, p, f = np.full((3, cols + forecast_period), np.nan)
    q = 1 #periods since last demand observation

    # Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0] / p[0] # Create all the t+1 forecasts
    for t in range(0,cols):
        if d[t] > 0:
            a[t +1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            f[t + 1] = a[t + 1] / p[t + 1]
            q = 1
        else:
            a[t + 1] = a[t]
            p[t + 1] = p[t]
            f[t + 1] = f[t]
            q += 1

    # Future Forecast
    a[cols + 1 : cols + forecast_period] = a[cols]
    p[cols + 1 : cols + forecast_period] = p[cols]
    f[cols + 1 : cols + forecast_period] = f[cols]

    rmse = calculate_rmse('Croston', d[0:cols - 1], f[0:cols - 1])
#    plotting('Croston', f[0 : cols - 1], d[0 : cols - 1])
    forecast = f[cols : cols + forecast_period]

    return rmse,forecast

#%%
def Croston_TSB(dataset, forecast_period = 1, alpha = 0.4, beta = 0.4):
    rmse=dict()
    pred_croston=dict()
    rmse_val=[]
    d = np.array(dataset) # Transform the input into a numpy array
    cols = len(d) # Historical period length
    d = np.append(d, [np.nan] * forecast_period) # Append np.nan into the demand array to cover future periods

    #level (a), probability(p) and forecast (f)
    a, p, f = np.full((3, cols + forecast_period), np.nan)# Initialization
    first_occurence = np.argmax(d[:cols] > 0)
    a[0] = d[first_occurence]
    p[0] = 1/(1 + first_occurence)
    f[0] = p[0] * a[0]

    # Create all the t+1 forecasts
    for t in range(0, cols):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = beta * (1) + (1 - beta) * p[t]
        else:
            a[t + 1] = a[t]
            p[t + 1] = (1 - beta) * p[t]
        f[t + 1] = p[t + 1] * a[t + 1]

    # Future Forecast
    a[cols + 1 : cols + forecast_period] = a[cols]
    p[cols + 1 : cols + forecast_period] = p[cols]
    f[cols + 1 : cols + forecast_period] = f[cols]

    rmse_val.append(calculate_rmse('Croston', d[0:cols - 1], f[0:cols - 1]))
    rmse['Croston']=rmse_val
    plotting('Croston', f[0 : cols - 1], d[0 : cols - 1])
    forecast = f[cols : cols + forecast_period]
    forecast = [int(i) for i in forecast]
    pred_croston['Croston']=forecast
    return rmse, pred_croston
