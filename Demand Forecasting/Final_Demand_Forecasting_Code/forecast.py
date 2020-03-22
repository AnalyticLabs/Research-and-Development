# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 00:48:58 2019

@author: AKE9KOR
"""

#%%
import pandas as pd
import numpy as np
import math
import json
#Importing libraries
import model_config as mc
import modeling_util as mu
from statistics import variance
import ml_algo as ml
import ts_algo as ts
import outputs as op
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#%%
def model_predict(best_algo, best_order, data, forecast_period,rept=0):

    predictions = []
    ML_models = mc.init_ML_hyp_models()
    ARIMA_models = mc.init_ARIMA_models(best_order)
    param_val={}
    if best_algo in ML_models.keys():
        if best_algo == 'GR_LR':
            predictions = ml.model_LinearRegression(data.values, forecast_period, best_order)
        elif best_algo == 'SVR_Sigmoid':
            predictions = ml.model_SVR_Sigmoid(data.values, forecast_period, best_order)
        elif best_algo == 'SVR_RBF':
            predictions = ml.model_SVR_RBF(data.values, forecast_period, best_order)
        else:
            test_shape_add=mc.test_shape_adder(best_algo)
            test_shape_fin=forecast_period+test_shape_add
            predictions, rmse,param_val = ml.model_ML(dataset = data.values, tsize = forecast_period, test_shape = test_shape_fin, model = ML_models[best_algo], order = best_order)
            if test_shape_add > 0:
                st=test_shape_fin-1
                end=test_shape_add-1
                predictions=predictions[-st:-end]
#            predictions=pd.DataFrame(predictions)
#            predictions=predictions.head(10)

#            end=test_shape_fin-forecast_period
#            if test_shape_fin-forecast_period==0:
#                 predictions=predictions[-test_shape_fin:]
#            else:
#                st=forecast_period + test_shape_add
#                end=test_shape_add


    elif best_algo in ARIMA_models.keys():
#        print("lolllllllzzz",ARIMA_models[best_algo])
        predictions , rmse = ts.model_ARIMA(best_algo,train = data.values, test_shape = forecast_period, order = ARIMA_models[best_algo],train_flag=0)

    elif best_algo in ['SES', 'HWES']:
        predictions, rmse = ts.model_ES(best_algo, train = data.values, test_shape = forecast_period,train_flag=0)
    elif best_algo in ['naive', 'naive2','naive3','naive6', 'naive12', 'naive12wa','naive6wa']:
        predictions, rmse = ts.model_Naive(best_algo, data.values, forecast_period, best_order,rept,train_flag=0)

    elif best_algo in ['sma', 'wma']:
        predictions, rmse = ts.model_MA(best_algo, data.values, forecast_period, train_flag = 0)
    else:
        predictions = [0]*forecast_period

    return predictions,param_val

#%%
def find_best_model(outputs):
    best_models = dict()
    for element in outputs:
        sku = element['sku']
        model = element['best_model']
        best_models[sku] = model
    return best_models


#%%
def find_intervals(outputs):
    intervals = dict()
    for element in outputs:
        sku = element['sku']
        interval = element['interval']
        intervals[sku] = interval
    return intervals


#%%
def find_forecast_period(outputs, interval):

    pass

#%%
def plot_forecast(dataset, forecast, color):
    plt.figure(figsize=(30, 7))
    plt.plot(dataset.index, dataset['sales'], 'b-')
    a = len(dataset)
    b = len(forecast)
    index = range(a - 1, a + b - 1)
    plt.plot(index, forecast, color)
    plt.show()


#%%
def plot_all_forecasts(dataset, data, forecast, forecast_en, forecast_ml, forecast_ts, sku):
    dataset = dataset.reset_index()
#    data = data.reset_index()
    dataset.columns = ['time', 'sales']
#    data.columns = ['time', 'sales']
#    last_date = dataset.time.iloc[-1]
    plt.figure(figsize=(30, 7))
    plt.title(str(sku))
    plt.plot(dataset.index, dataset['sales'], 'y-')
    plt.plot(data, 'b-')
    a = len(dataset)
    b = len(forecast)
    index = range(a - 1, a + b - 1)
    plt.plot(index, forecast, 'k')
    plt.plot(index, forecast_ml, 'r')
    plt.plot(index, forecast_ts, 'c')
    plt.plot(index, forecast_en, 'g')

#    plt.savefig(r'C:\Users\INE4KOR\Desktop\Outputs\sku_' + str(sku) + '.png')
    plt.show()

#%%
def output_forecast(sku, dataset, sku_data, output, forecast_results):
    dataset = dataset.reset_index()
    dataset.columns = ['time', 'sales']

    output['last_date'] = dataset.time.iloc[-1]

    forecast_result = op.add_forecasted_results(sku, dataset, sku_data, output)
    forecast_results.append(forecast_result)

    with open(r'C:\Users\INE4KOR\Desktop\FD_all.json', 'w', encoding = 'utf-8') as f:
        json.dump(forecast_results, f, ensure_ascii = False, indent = 4, default = str)

    return forecast_results
#%%
def calculate_r2(expected,forecast):
    if math.isnan(expected):
        expected = 0
    else:
        expected = int(expected)
    sample=[expected,forecast]
    variance=np.var(sample)
    expected=[expected]
    forecast=[forecast]
    mse=mean_squared_error(expected,forecast)
    rsquare=1-(mse/variance)
#    rsquare=r2_score(expected, forecast)*100
    return rsquare
#%%
def calculate_forecast_accuracy(expected, forecast):
    if math.isnan(expected):
        expected = 0
    else:
        expected = int(expected)
    expected = int(expected)
    forecast = int(forecast)
    print("calculate_forecast_accuracy")
    if expected != 0:
        facc = (1 - (np.abs(expected - forecast)) / (expected)) * 100
        mape = (np.abs(expected - forecast) / expected) * 100
    elif forecast!=0:
        facc = (1 - (np.abs(expected - forecast)) / (forecast)) * 100
        mape = (np.abs(expected - forecast) / forecast) * 100
    elif expected==0 and forecast==0:
        facc=100
        mape=0

    if facc<0:
        facc=0

    bias = (expected - forecast)

    if np.isnan(facc)== True or np.isfinite(facc)==False:
        facc=0
    if np.isnan(mape)== True or np.isfinite(mape)==False:
        mape=0
    if np.isnan(bias)== True or np.isfinite(bias)==False:
        mape=0

    return float(format(facc,'.3f')), float(format(mape,'.3f')), float(format(bias,'.3f'))

#%%
def calculate_validation_facc(expected, predictions):
    validation_facc = []
    for i in range(len(expected)):
        a = int(expected[i])
        b = int(predictions[i])
        if a != 0:
            value = ((1 - (np.abs(a - b)) /(a)) * 100)
        elif b!=0:
            value = (1 - (np.abs(a - b)) / (b)) * 100
        elif a==0 and b==0:
            value=100
        if np.isnan(value)== True or np.isfinite(value)==False:
            value=0
        validation_facc.append(float(format(value,'.3f')))

    validation_facc = [0 if i<0 else i for i in validation_facc]
    print("Validation Accuracy")
    print(validation_facc)
    return validation_facc



