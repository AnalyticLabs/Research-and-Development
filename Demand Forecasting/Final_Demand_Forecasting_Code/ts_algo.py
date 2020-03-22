# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:43:21 2019

@author: AKE9KOR
"""
#%%
#Import libraries
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
import util as ut
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import modeling_util as mu

#%%
'''
Function to evaluate an ARIMA model for a given order (p, d, q)

'''

def evaluate_arima_model(X, arima_order):
    #Prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    #Evaluate ARIMA model
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp = 0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(yhat)

    #Measure error using RMSE
    error = sqrt(mean_squared_error(test, predictions))
    return error

#%%
'''

'''
def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)

                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order

                except:
                    continue

    return best_cfg
#%%
def model_ARMA(key,train, test_shape, order, train_flag = 0, test = []):
    predictions = []
    rmse_val=[]
    if(train_flag):
        test=test[0]
    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                for j in range(3):
                    model = ARMA(v_train, order = order)
                    model_fit = model.fit(disp=0, transparams=False, trend='nc')
                    yhat = model_fit.forecast()[0]
                    pred = yhat
                    if pred < 0:
                        pred = mu.weighted_moving_average(v_train, 1, 3)
                        pred = pred[0]
                    pred_temp.append(pred)
                    v_train.append(pred)

            except:
                pred_temp.extend(mu.moving_average(v_train, 3 - len(pred_temp), 3))
            mu.plotting(key, pred_temp, v_expected)
            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            rmse_val.append(mu.calculate_rmse(key, v_expected, pred_temp))
            itr=itr-1


    else:
        try:
            for t in range(test_shape):
                model = ARMA(history, order = order)
                model_fit = model.fit(disp=0, transparams=False, trend='nc')
                yhat = model_fit.forecast()[0]
                inverted = list()
                for i in range(len(yhat)):
                    value = ut.inverse_difference(history, yhat[i], len(history) - i)
                    inverted.append(value)
                inverted = np.array(inverted)
                pred = inverted[-1]
                if pred < 0:
                    pred = mu.weighted_moving_average(history, 1, 3)
                predictions.append(pred)
                history.append(yhat)
        except:
            predictions.extend(mu.moving_average(history, test_shape - len(predictions), 3))

    predictions = [int(i) for i in predictions]
    return predictions,rmse_val
#%%
def model_ARIMA(key,train, test_shape, order, train_flag = 0, test = []):
    predictions = []
    rmse_val = []
    if(train_flag):
        test=test[0]
    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                order=(order[0],1,order[2])
                for j in range(3):
                    model = ARIMA(v_train, order = order)
                    model_fit = model.fit(disp=0)
                    yhat = model_fit.forecast()[0]
                    if yhat < 0:
                        yhat= mu.weighted_moving_average(history,1,3)
                        yhat=yhat[0]
                    pred_temp.append(yhat)
                    v_train.append(yhat)
            except:
                pred_temp.extend(mu.moving_average(v_train, 3 - len(pred_temp), 3))
            mu.plotting(key, pred_temp, v_expected)
            rmse_val.append(mu.calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            itr=itr-1
    else:
        try:
            #TODO:check order
            order=(order[0],1,order[2])
            for t in range(test_shape):
                model = ARIMA(history, order = order)
                model_fit = model.fit(disp=0)
                yhat = model_fit.forecast()[0]
                if yhat < 0:
                    yhat= mu.weighted_moving_average(history,1,3)
                yhat=yhat[0]
                predictions.append(yhat)
                history.append(yhat)
        except:
                predictions.extend(mu.moving_average(history, test_shape - len(predictions), 3))

    predictions = [0 if pd.isnull(i) else int(i) for i in predictions]
    return predictions,rmse_val

#%%
def model_ES(key, train, test_shape = 0, train_flag = 0, test = []):
    predictions = []
    rmse_val=[]

    try:
        train = train.values
    except:
        train = train
    history = [np.asscalar(x) for x in train]

#   TRAIN
    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=[np.asscalar(x) for x in data[:-itr].values]
            v_expected=data.tail(itr).head(3).reset_index(drop = True)
            try:
                for t in range(3):
                    if key=='SES':
                        model = SimpleExpSmoothing(history)
                    elif key=='HWES':
                         model = ExponentialSmoothing(history)
                    model_fit = model.fit()
                    yhat= model_fit.predict(len(history), len(history))
                    if yhat < 0:
                        yhat= mu.weighted_moving_average(history,1,3)
                    yhat=yhat[0]
                    pred_temp.append(yhat)
                    v_train.append(yhat)
            except:
                pred_temp.extend(mu.moving_average(v_train, 3 - len(pred_temp), 3))
            mu.plotting(key, pred_temp, v_expected)
            rmse_val.append(mu.calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            itr=itr-1
#   FORECAST
    else:
        try:
            for t in range(test_shape):
                if key=='SES':
                    model = SimpleExpSmoothing(history)
                elif key=='HWES':
                     model = ExponentialSmoothing(history)
                model_fit = model.fit()
                yhat= model_fit.predict(len(history), len(history))
                if yhat < 0:
                    yhat= mu.weighted_moving_average(history,1,3)
                yhat=yhat[0]
                predictions.append(yhat)
                history.append(yhat)
        except:
            predictions.extend(mu.moving_average(history, test_shape - len(predictions), 3))

    predictions = [int(i) for i in predictions]
    return predictions,rmse_val

#%%
def model_Naive(key,train,test_shape,order,rept,train_flag = 0):
   forecast = []
   p=order[0]
   rmse_val=[]
   try:
       train = train.values
   except:
       train = train

   history = [np.asscalar(x) for x in train]
   if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=data[:-itr]
            v_train=v_train.values
            v_expected=data.tail(itr).head(3)

            for j in range(3):

                if key =='naive':
                    try:
                        t = v_train[-p]
                    except:
                        t=0

                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive_rept':
                    try:
                        t = v_train[-rept]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive3':
                    try:
                        t = v_train[-3]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive6':
                    try:
                        t = v_train[-6]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive12':
                    try:
                        t=v_train[-12]
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key == 'naive12wa':
                    try:
                        yt=v_train[-12]
                    except:
                        yt=0
                    try:
                        yt_1=v_train[-24]
                    except:
                        yt_1=0
                    t = ((0.55*yt)+(0.45*yt_1))
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)
                elif key=='naive6wa':
                    try:
                        #naive of six
                        try:
                            naive_six=v_train[-6]
                        except:
                            naive_six=0
                        #weighted moving average
                        alpha=[0.25,0.35,0.4]
                        pred1 = v_train[-3:]
                        pred1=[np.asscalar(x) for x in pred1]
                        weighted_avg=np.dot(pred1,alpha)
                        #ensemble
                        t=(0.6*naive_six)+(0.4*weighted_avg)
                    except:
                        t=0
                    pred_temp.append(t)
                    v_train=np.append(v_train,t)


            rmse_val.append(mu.calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                forecast.extend(pred_temp)
            else:
                forecast.append(pred_temp[0])

            itr=itr-1
   else:
        for num in range(test_shape):
           if key =='naive':
               try:
                   t = history[-p]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive2':
               try:
                   t = history[-rept]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive3':
               try:
                   t = history[-3]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive6':
               try:
                   t = history[-6]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive12':
               try:
                   t = history[-12]
                   forecast.append(t)
                   history.append(t)
               except:
                   pass
           elif key == 'naive12wa':
               try:
                   yt=history[-12]
               except:
                   yt=0
               try:
                   yt_1=history[-24]
               except:
                   yt_1=0
               t = ((0.55*yt)+(0.45*yt_1))
               forecast.append(t)
               history.append(t)
           elif key=='naive6wa':
               #naive of six
               try:
                   naive_six=history[-6]
               except:
                   naive_six=0
                #weighted moving average
               alpha=[0.25,0.35,0.4]
               pred1 = history[-3:]
#               pred1=[np.asscalar(x) for x in pred1]
               weighted_avg=np.dot(pred1,alpha)
               #ensemble
               t=(0.6*naive_six)+(0.4*weighted_avg)
               forecast.append(t)
               history.append(t)

   forecast = [int(i) for i in forecast]
   return forecast,rmse_val

#%%
def model_MA(key,train,test_shape,train_flag = 0):
    forecast = []
    rmse_val=[]
    try:
       train = train.values
    except:
       train = train
    history = [np.asscalar(x) for x in train]

# TRAIN
    if train_flag==1:
        itr=5
        data=pd.DataFrame(history)
        for i in range(3):
            pred_temp = []
            v_train=data[:-itr]
            v_train=v_train.values
            v_expected=data.tail(itr).head(3)
            for j in range(3):
                if key =='sma':
                    pred1 = np.mean(v_train[-3:])
                    pred_temp.append(pred1)
                    v_train=np.append(v_train,pred1)
                if key =='wma':
                    alpha=[0.25,0.35,0.4]
                    pred1 = v_train[-3:]
                    pred1=[np.asscalar(x) for x in pred1]
                    pred1=np.dot(pred1,alpha)
                    pred_temp.append(pred1)
                    v_train=np.append(v_train,pred1)
            rmse_val.append(mu.calculate_rmse(key, v_expected, pred_temp))

            if i == 2:
                forecast.extend(pred_temp)
            else:
                forecast.append(pred_temp[0])

            itr=itr-1
#   FORECAST
    else:

        for num in range(test_shape):
            if key =='sma':
                test_new = pd.DataFrame(history)
                pred1 = (test_new.tail(3).mean())
                pred1 = pred1[0]
                forecast.append(pred1)
                history.append(pred1)
            if key =='wma':
                alpha=[0.25,0.35,0.4]
                test_new = pd.DataFrame(history)
                pred1 = test_new.tail(3)
                pred1=np.dot(pred1[0],alpha)
                forecast.append(pred1)
                history.append(pred1)

    forecast = [int(i) for i in forecast]
    return forecast,rmse_val