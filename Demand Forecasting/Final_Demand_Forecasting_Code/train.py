# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:10:06 2019

@author: AKE9KOR
"""

#%%
#Importing libraries
from math import sqrt
from sklearn.metrics import mean_squared_error
import ml_algo as ml
import ts_algo as ts
import model_config as mc
import modeling_util as mu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
def time_series_using_ml(dataset, tsize, order,cluster):
    models = mc.models_ML(cluster)
#    models=mc.init_ML_models()
    rmse = dict()
    model_predictions = dict()
#    mape=dict()
#    itr=5
    for key in models.keys():
        test_shape = tsize

        test_shape_incr = mc.init_test_shape()
        if key in test_shape_incr:
            test_shape = test_shape + test_shape_incr[key]
        else:
            test_shape = test_shape + 1

        predictions,rmse_i,param_val = ml.model_ML(dataset.values, tsize, test_shape, models[key], key, order, 1)
        predictions=predictions[-tsize:]
#        print("before",len(predictions))
#        if test_shape_incr[key] > 0:
#                st=test_shape-1
#                end=test_shape_incr[key]-1
#                predictions=predictions[-st:-end]
#        else:
#            predictions=predictions[-tsize:]
#        if key in test_shape_incr:
#            for i in range(0, test_shape_incr[key]):
#                predictions.pop(0)
#        else:
#            predictions.pop(0)
#        print("after",len(predictions))
#        rmse[key] = mu.calculate_rmse(key, expected, predictions)
#        mu.plotting(key, predictions, expected)
#        print("mape",mape)
        rmse[key]=rmse_i
        model_predictions[key] = predictions

#    ####### Models
#    expected=dataset.tail(5)
#    model_predictions['GR_LR'], rmse['GR_LR'] = ml.model_LinearRegression(dataset.values, tsize, order, 1)
#    rmse['GR_LR'] = mu.calculate_rmse('GR_LR', expected, predictions)
#    mu.plotting('GR_LR', predictions, expected)
#
#    model_predictions['SVR_Sigmoid'] = ml.model_SVR_Sigmoid(dataset.values, tsize, order, 1)
#    rmse['SVR_Sigmoid'] = mu.calculate_rmse('SVR_Sigmoid', expected, predictions)
#    mu.plotting('SVR_Sigmoid', predictions, expected)
#
#    model_predictions['SVR_RBF'] = ml.model_SVR_RBF(dataset.values, tsize, order, 1)
#    rmse['SVR_RBF'] = mu.calculate_rmse('SVR_RBF', expected, predictions)
#    mu.plotting('SVR_RBF', predictions, expected)
#
##    model_predictions['SVR_Poly'] = ml.model_SVR_Poly(dataset.values, tsize, order, 1)
##    rmse['SVR_Poly'] = mu.calculate_rmse('SVR_Poly', expected, predictions)
##    mu.plotting('SVR_Poly', predictions, expected)
#
#    model_predictions['GR_DT'] = ml.model_DecisionTree(dataset.values, tsize, order, 1)
#    rmse['GR_DT'] = mu.calculate_rmse('GR_DT', expected, predictions)
#    mu.plotting('GR_DT', predictions, expected)
#
#    model_predictions['GR_RF'] = ml.model_RandomForest(dataset.values, tsize, order, 1)
#    rmse['GR_RF'] = mu.calculate_rmse('GR_RF', expected, predictions)
#    mu.plotting('GR_RF', predictions, expected)
#
#    model_predictions['GR_Ridge'] = ml.model_Ridge(dataset.values, tsize, order, 1)
#    rmse['GR_Ridge'] = mu.calculate_rmse('GR_Ridge', expected, predictions)
#    mu.plotting('GR_Ridge', predictions, expected)
#
#    model_predictions['GR_Lasso'] = ml.model_Lasso(dataset.values, tsize, order, 1)
#    rmse['GR_Lasso'] = mu.calculate_rmse('GR_Lasso', expected, predictions)
#    mu.plotting('GR_Lasso', predictions, expected)
#
#    model_predictions['GR_ElasticNet'] = ml.model_ElasticNet(dataset.values, tsize, order, 1)
#    rmse['GR_ElasticNet'] = mu.calculate_rmse('GR_ElasticNet', expected, predictions)
#    mu.plotting('GR_ElasticNet', predictions, expected)


    return model_predictions,rmse

#%%
def run_arima_models(data, diff_data, best_order, tsize,cluster):

    models=mc.models_arima(best_order,cluster)
#    models = mc.init_ARIMA_models(best_order)

    rmse=dict()
    model_predictions = dict()

    train, test = data[0:-tsize], data[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
    expected = test
    test_shape = len(test)
    for key in models.keys():
        print("KEY!", key)
        if key == 'ARMA' or key == 'AR' or key == 'MA':
            predictions,rmse_i = ts.model_ARMA(key,train, test_shape, models[key], train_flag = 1, test= test)

        else:
            if key == 'ARIMA':
                predictions,rmse_i = ts.model_ARIMA(key,train, test_shape, models[key], train_flag = 1, test= test)

#        mu.plotting(key, predictions, expected)
        rmse[key]=rmse_i

        model_predictions[key] = predictions


    return rmse, model_predictions
#%%
def naive_forecast(dataset,freq,p,tsize):
    rmse = dict()
    model_predictions = dict()
    train, test = dataset[0:-tsize], dataset[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
#    expected = test
    test_shape = len(test)
    sam=np.array(dataset)
    repeat=mu.check_repetition(sam, freq , 1, len(sam))

    key='naive'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive_rept'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive3'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive6'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive12'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions

    key='naive12wa'
    predictions,rmse_i = ts.model_Naive(key,train, test_shape, p, repeat, train_flag = 1)
    rmse[key]=rmse_i
    model_predictions[key] = predictions



    return rmse,model_predictions


#%%
def run_es_models(train, test,cluster):
    key=mc.models_ES(cluster)
    rmse = dict()
    test_shape = len(test)
#    expected = test
    model_predictions = dict()
#    models = mc.init_ES_models()

#    key='SES'
#    predictions,rmse_i = ts.model_ES(key, train, test_shape, train_flag = 1, test = test)
#    rmse[key]=rmse_i
#    model_predictions[key] = predictions
#
#    key='HWES'
#    predictions,rmse_i = ts.model_ES(key, train, test_shape, train_flag = 1, test = test)
#    rmse[key]=rmse_i
#    model_predictions[key] = predictions
#    mu.plotting(key, predictions, expected)
    if len(key)>0:
        if len(key)==2:
            #SES
            predictions,rmse_i = ts.model_ES(key[0], train, test_shape, train_flag = 1, test = test)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions
            #HWES
            predictions,rmse_i = ts.model_ES(key[1], train, test_shape, train_flag = 1, test = test)
            rmse[key[1]]=rmse_i
            model_predictions[key[1]] = predictions

        else:
            predictions,rmse_i = ts.model_ES(key[0], train, test_shape, train_flag = 1, test = test)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

            model_predictions[key[0]] = predictions



    return rmse, model_predictions
#%%
def Moving_Average(data,tsize,cluster):
    rmse = dict()
    model_predictions = dict()
    key=mc.models_Averages(cluster)
    train, test = data[0:-tsize], data[-tsize:]
    test=pd.DataFrame(test)
    test=test.reset_index(drop=True)
#    expected = test
    test_shape = len(test)
    if len(key)>0:
        if len(key)==2:
            predictions,rmse_i = ts.model_MA(key[0],train, test_shape, train_flag = 1)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

            predictions,rmse_i = ts.model_MA(key[1],train, test_shape, train_flag = 1)
            rmse[key[1]]=rmse_i
            model_predictions[key[1]] = predictions
        else:
            predictions,rmse_i = ts.model_MA(key[0],train, test_shape, train_flag = 1)
            rmse[key[0]]=rmse_i
            model_predictions[key[0]] = predictions

    print("Moving_Average done")

    return rmse,model_predictions

#%%
def time_series_models(freq,data, diff_data, tsize, best_order,cluster):
    rmse_ARIMA, predictions_ARIMA = run_arima_models(data, diff_data, best_order, tsize,cluster)
    rmse_ES, predictions_ES = run_es_models(data, diff_data,cluster)
    rmse_naive, predictions_naive = naive_forecast(data,freq,best_order,tsize)
    rmse_ma,predictions_ma = Moving_Average(data,tsize,cluster)
    return rmse_ARIMA, rmse_ES, rmse_naive, rmse_ma, predictions_ARIMA, predictions_ES, predictions_naive,predictions_ma

