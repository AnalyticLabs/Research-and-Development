# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:07:40 2019

@author: AKE9KOR
"""

#%%
#Importing libraries
import pandas as pd
import numpy as np
import scipy.stats as sst
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import forecast as ft
import modeling_util as mu


#%%
def weight_calculation(data,best_models,best_order):
    itr=5
    weight_ts=0
    weight_ml=0
    for i in range(3):
        print("Running models for ensemble ...",i)
        sample=data[:-itr]
        expected=data.tail(itr).head(3)
        forecast_ml,p = ft.model_predict(best_models[0], best_order, sample, 3)
        forecast_ts,p= ft.model_predict(best_models[1], best_order, sample, 3)
        itr-=1
        expected=expected.reset_index(drop=True)
        forecast_ts=pd.DataFrame(forecast_ts)
        rmse_ts=mu.calculate_rmse(best_models[1], expected, forecast_ts)
        rmse_ml=mu.calculate_rmse(best_models[0], expected, forecast_ml)
        weight_ts+=calculate_weight(rmse_ts,rmse_ml)
        weight_ml+=calculate_weight(rmse_ml,rmse_ts)
    weight_ts=weight_ts/3
    weight_ml=weight_ml/3
    return weight_ts,weight_ml
#%%
#weights for  ensemble method
def calculate_weight(error1,error2):
    if error1 == 0.0:
        a = 1
        b = 0
    elif error2 == 0.0:
        b =1
        a=0
    else:
        a=1/error1
        b=1/error2
    weight=a/(a+b)
    return weight


#%%
def method_ensemble(forecast_ml,forecast_ts, weight_ml, weight_ts,forecast_period):

    forecast=[0]*forecast_period

    for i in range(forecast_period):
        forecast[i]=((weight_ml*forecast_ml[i])+(weight_ts*forecast_ts[i]))/(weight_ml+weight_ts)
        forecast = [0 if i < 0 else int(i) for i in forecast]

    return forecast
#%%
'''
Create a differenced series
dataset                 --> data for a sku
interval                --> difference between the two indices
value                   --> difference between values of two indices in datset
diff                    --> list of differenced values
'''
def difference(dataset, interval = 1):
    diff = list()
    if(interval != 0):
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
    else:
        diff = list(dataset)
#        for num in dataset:
#            diff.append(np.asscalar(num))
#        print(diff)

    return pd.Series(diff)

#%%
'''
Convert sequence to a supervised learning problem
data                  -->
lag                   -->
dataset               --> data for a sku, converted to a dataframe
columns               -->
'''
def timeseries_to_supervised(dataset, lag =1):
    dataset = pd.DataFrame(dataset)
    y = [dataset.shift(i) for i in range(1, lag + 1)]
    y.append(dataset)
    dataset = pd.concat(y, axis = 1)
    cols = []
    for i in range(lag):
        cols.append('x_' + str(i))
    cols.append('y')
    dataset.columns = cols
#    print(dataset.columns)
    dataset.dropna(axis=0,inplace = True)
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back):
#        a = dataset[i:(i + look_back),0]
#        dataX.append(a)
#        dataY.append(dataset[i + look_back,0])
#    return np.array(dataX), np.array(dataY)
    return dataset
#%%
def scaler_selection(key):
    if key == 'lr' or key == 'lasso' or key == 'ridge' or key == 'knn':
        scaler = MinMaxScaler(feature_range=(0,1), copy=True)
    elif key == 'svmr':
        scaler = StandardScaler()

    return scaler
#%%
'''
Invert differenced values
history              -->
yhat                 -->
interval             -->
'''
def inverse_difference(history, yhat, interval = 1):
    return yhat + history[-interval]
#%%

def seasonal_effect(forecast, ft_p, stype, seasonality, j):
    for i in range(ft_p):
        if stype=='additive':
            forecast[i]=forecast[i]+seasonality[j]
        elif stype=='multiplicative':
            forecast[i]=forecast[i]*seasonality[j]
            j+=1
            if j>=len(seasonality)-1:
                j=0
    return forecast
#%%
def init_output(forecast_period, raw_data,prof):
    output = {}
    insight={}
    insight['Seasonality']=prof.Seasonal
    insight['Trend_Type']=prof.Trend
    insight['Sparsity']=prof.spar
    insight['NPI']=prof.NPI
    output['insights']=insight
    output['forecast_period'] = forecast_period
    output['forecast_values'] = []
    output['interval'] = 'M'
    output['actuals'] = assign_dates(raw_data, 'actuals')
    output['best_models_ml'] = []
    output['best_models_ts'] = []
    output['bias_ml'] = []
    output['bias_ts'] = []
    output['bias_en'] = []
    output['accuracy_ml'] = []
    output['accuracy_ts'] = []
    output['accuracy_en'] = []
    output['validation'] = dict()
    output['validation_facc'] = dict()
    output['facc'] = ''
    output['mape'] = ''
    output['bias'] = ''
    output['TS']=dict()
    output['ML']=dict()
    output['Ensemble']=dict()
    output['forecast_ml'] = []
    output['forecast_ts'] = []
    output['forecast_en'] = []
    output['model_ml'] = ''
    output['model_ts'] = ''

    return output

#%%
def assign_dates(data, flag, dates = ''):
    if flag == 'validation':
        dates = dates.reset_index()
        dates.columns = ['time', 'sales']
        dates.time = pd.to_datetime(dates.time, format = '%d/%m/%y', infer_datetime_format = True)
        dates.time = dates.time.dt.to_period('M')
        data=[float(format(i,'.3f')) for i in data]
        result = pd.DataFrame({'time': dates.time.astype(str), 'validation': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['validation']

    elif flag == 'val_facc':
        dates = dates.reset_index()
        dates.columns = ['time', 'sales']
        dates.time = pd.to_datetime(dates.time, format = '%d/%m/%y', infer_datetime_format = True)
        dates.time = dates.time.dt.to_period('M')
        result = pd.DataFrame({'time': dates.time.astype(str), 'val_facc': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['val_facc']


    elif flag == 'forecast':
        dates = dates.reset_index()
        dates.columns = ['time', 'forecast']
        last_date = pd.to_datetime(dates.time[0], format = '%d/%m/%y', infer_datetime_format = True)
        print(last_date)
        date_range = pd.date_range(last_date, periods = 6, freq = 'M')
        date_range = date_range.strftime('%Y-%m').tolist()
        date_range.pop(0) #same month as last_date, FUSO

        data=[float(format(i,'.3f')) for i in data]
        result = pd.DataFrame({'time': date_range, 'forecast': data})
        result.set_index('time', inplace = True)
        result = result.to_dict()
        result = result['forecast']


    elif flag == 'actuals':
        data = data.tail(8).reset_index()
        data.columns = ['time', 'sales']
        data['time'] = pd.to_datetime(data['time'], format = '%d/%m/%y', infer_datetime_format = True)
        data['time'] = data['time'].dt.to_period('M').astype(str)
        data['sales'] = [0 if pd.isnull(i) else int(i) for i in data['sales']]
        data['sales']=[float(format(i,'.3f')) for i in data['sales']]
        data.set_index('time', inplace = True)
        result = data.to_dict()
        result = result['sales']



    return result