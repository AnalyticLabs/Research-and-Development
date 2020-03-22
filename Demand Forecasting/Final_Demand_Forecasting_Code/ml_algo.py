# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:42:42 2019

@author: AKE9KOR
"""

#%%
#Import libraries
import numpy as np
import pandas as pd
from math import sqrt
import warnings
import copy

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR
from scipy.stats import randint as sp_randint
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

#project related libaries
import util as ut
import modeling_util as mu
import model_config as mc
#import feature_selection as fs
warnings.filterwarnings('ignore')
#%%
def model_ML(dataset = [], tsize = 0, test_shape = 0, model = np.nan, key='', order = (0, 0, 0), train_flag = 0):
    predictions = []
    pred_temp = []
    rmse_val=[]
    parameter_values={}
    scale_flag=0
    if key == 'lr' or key == 'lasso' or key == 'ridge' or key == 'knn' or key == 'svmr':
        scale_flag=1

    if train_flag==1:
        itr=5
        for i in range(3):
            expected=pd.DataFrame(dataset)
            expected=expected.tail(itr).head(3)
            expected=expected.reset_index(drop=True)

            train=dataset[:-itr]

            diff_values = ut.difference(train, order[1])

            if scale_flag==1:
                scaler=ut.scaler_selection(key)
                diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

            supervised= ut.timeseries_to_supervised(train, order[0])
            data=supervised.values

            RF_model = mu.fit_model(data, model)


            pred_temp=[]

            for j in range(test_shape):
                X, y = data[:, 0:-1], data[:, -1]
                yhat = mu.forecast_model(RF_model, X)

#TODO: Inverse differencing and scaling

#                if scale_flag==1:
#                    yhat=scaler.inverse_transform(pd.DataFrame(yhat).values.reshape(-1,1))
#                if order[1]!=0:
#                    inverted = list()
#                    for i in range(len(yhat)):
#                        value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
#                        inverted.append(value)
#                    inverted = np.array(inverted)
#                    forecast=inverted[-1]
#                else:
#                    forecast = yhat[-1]
                forecast=yhat[-1]
                if forecast < 0:
                    forecast = mu.weighted_moving_average(dataset, 1, 3)[0]

                pred_temp.append(forecast)

                train = np.append(train, forecast)

                diff_train = ut.difference(train, order[1])

                if scale_flag==1:
                    scaler=ut.scaler_selection(key)
                    diff_train=scaler.fit_transform(pd.DataFrame(diff_train).values.reshape(-1,1))

                supervised= ut.timeseries_to_supervised(train, order[0])
                data=supervised.values

            pred_temp=pred_temp[1:4]
            mu.plotting(key, pred_temp, expected)
            if i == 2:
                predictions.extend(pred_temp)
            else:
                predictions.append(pred_temp[0])

            rmse_val.append(mu.calculate_rmse(key, expected, pred_temp))
            itr=itr-1

    else:

        dataset_1=copy.deepcopy(dataset)
        diff_values = ut.difference(dataset_1, order[1])

        if scale_flag==1:
            scaler=ut.scaler_selection(key)
            diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

        supervised= ut.timeseries_to_supervised(diff_values, order[0])
        data=supervised.values

        RF_model = mu.fit_model(data, model)
        try:
            parameter_values=model.best_params_
        except:
            parameter_values=model.get_params()

        test_shape=test_shape+2
        for i in range(test_shape):

            X, y = data[:, 0:-1], data[:, -1]

            yhat = mu.forecast_model(RF_model, X)
#
#            if scale_flag==1:
#                yhat=scaler.inverse_transform(pd.DataFrame(yhat).values.reshape(-1,1))
#            if order[1]!=0:
#                inverted = list()
#                for i in range(len(yhat)):
#                    value = ut.inverse_difference(data, yhat[i], len(data) - i)
#                    inverted.append(value)
#                    inverted = np.array(inverted)
#                forecast=inverted[-1]
#            else:
#                forecast = yhat[-1]
            forecast=yhat[-1]
            if forecast < 0:
                forecast = mu.weighted_moving_average(data, 1, 3)[0]

            predictions.append(forecast)
            dataset_1 = np.append(dataset_1, forecast)

            diff_values = ut.difference(dataset_1, order[1])

            if scale_flag==1:
                scaler=ut.scaler_selection(key)
                diff_values=scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1,1))

            supervised= ut.timeseries_to_supervised(diff_values, order[0])
            data=supervised.values
        predictions=predictions[2:test_shape]
    predictions = [int(i) for i in predictions]
    return predictions,rmse_val,parameter_values


#%%
def model_LinearRegression(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    rmse_val = []


    if train_flag == 1:
        itr = 5
        for i in range(3):
            expected = pd.DataFrame(dataset)
            expected = expected.tail(itr).head(3).reset_index(drop = True)

            train = dataset[:-itr]
            diff_values = ut.difference(dataset, order[1])

            scaler = ut.scaler_selection('lr')
            diff_values = scaler.fit_transform(pd.DataFrame(diff_values).values.reshape(-1, 1))

            supervised = ut.timeseries_to_supervised(diff_values, order[0])
            data = supervised.values

            clf = LinearRegression()
            param = {"fit_intercept": [True, False],
			     "normalize": [False],
			     "copy_X": [True, False]}
            grid = GridSearchCV(clf, param, n_jobs = 1)
            model = mu.fit_model(data, grid)

            for j in range(tsize):
                X, y = data[:, 0:-1], data[:, -1]
                yhat = mu.forecast_model(model, X)

#    inverted = list()
#    for i in range(len(yhat)):
#        value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
#        inverted.append(value)
#    inverted = np.array(inverted)

                forecast = yhat[-1]
                if forecast < 0:
                    forecast = mu.weighted_moving_average(dataset, 1, 3)[0]

                predictions.append(forecast)
                train = np.append(train, forecast)
                diff_train = ut.difference(train, order[1])
                diff_train = scaler.fit_transform(pd.DataFrame(diff_train).values.reshape(-1, 1))

                supervised = ut.timeseries_to_supervised(train, order[0])
                data = supervised.values

            predictions = predictions[1:4]
            rmse_val.append(mu.calculate_rmse('GR_LR', expected, predictions))
            itr = itr - 1

    predictions = [int(i) for i in predictions]
    return predictions, rmse_val

#%%
def model_SVR_Sigmoid(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()
        g = list(np.linspace(0.0001,1,25000))
        C = [1]
        param = {"kernel": ["sigmoid"],
			     "gamma": g,
			     "C":C}
        random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        random_search.fit(X, y)
        clf = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

#%%
def model_SVR_RBF(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()

        g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]

        C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

        param= {'gamma': g,
			    'kernel': ['rbf'],
			    'C': C}
        grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        grid_search.fit(X, y)
        clf = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"])
        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

#%% SVR Poly
def model_SVR_Poly(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        mod = SVR()
        g = list(np.linspace(0.0001,1,1000))
        C = list(np.linspace(0.01,10,25))
        param = {"kernel": ["poly"],
		 	     "degree": range(10,30,1),
			     "gamma": g,
			     "C":C}
        random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
        random_search.fit(X, y)
        clf = SVR(kernel=random_search.best_params_["kernel"],degree=random_search.best_params_["degree"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])

        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

#%% Decision Tree
def model_DecisionTree(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]

        dtr = DecisionTreeRegressor()
        param_tree = {"max_depth": [3,None],
				  "min_samples_leaf": sp_randint(1, 11),
				  "criterion": ["mse"],
				  "splitter": ["best","random"],
				  "max_features": ["auto","sqrt",None]}

        gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
        gridDT.fit(X, y)
        clf = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])


        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

#%%
def model_RandomForest(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        rfr = RandomForestRegressor()
        param_forest = {"n_estimators": range(10,1000,100),
        				    "criterion": ["mse"],
        				    "bootstrap": [True, False],
        				    "warm_start": [True, False]
        			}
        gridRF = RandomizedSearchCV(rfr,param_forest,n_jobs=1,n_iter=100)
        gridRF.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions


#%%
def model_Ridge(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        rdg = Ridge()
        para_ridge = {"alpha": list(np.linspace(0.000000001,10000,1000000)),
				  "fit_intercept": [True, False],
				  "normalize": [True, False],
				  "solver": ["auto"]}
        random_rdg = RandomizedSearchCV(rdg, para_ridge, n_jobs=1, n_iter = 100)
        random_rdg.fit(X, y)
        clf = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions

#%%
def model_Lasso(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        lass = Lasso()
        param_lass = {"alpha": list(np.linspace(0.000000001,100,1000)),
				  "fit_intercept": [True, False],
				  "normalize": [True, False],
				  "selection": ["random"]}
        random_lass = RandomizedSearchCV(lass, param_lass, n_jobs=1, n_iter = 100)
        random_lass.fit(X, y)
        clf = Lasso(alpha=random_lass.best_params_["alpha"], fit_intercept=random_lass.best_params_["fit_intercept"],normalize=random_lass.best_params_["normalize"], selection= random_lass.best_params_["selection"])

        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions


#%%
def model_ElasticNet(dataset = [], tsize = 0, order = (0, 0, 0), train_flag = 0):
    predictions = []
    for i in range(tsize):
        diff_values = ut.difference(dataset, 1)
        supervised = ut.timeseries_to_supervised(diff_values, 1)
        data = supervised.values

        if train_flag == 1:
            train = data[0:-tsize]
        else:
            train = data

        X, y = train[:, 0:-1].reshape(-1, 1), train[:, -1]
        elas = ElasticNet()
        param = {"alpha": list(np.linspace(0.000000001,100,100000)),
			     "l1_ratio": list(np.linspace(0.000001,100,1000)),
			     "fit_intercept": [True, False],
			     "normalize": [True,False],
			     "precompute": [True, False]}
        random_elas = RandomizedSearchCV(elas, param, n_jobs=1, n_iter= 100)
        random_elas.fit(X, y)
        clf = ElasticNet(alpha = random_elas.best_params_["alpha"], l1_ratio= random_elas.best_params_["l1_ratio"], fit_intercept=random_elas.best_params_["fit_intercept"],
				      normalize=random_elas.best_params_["normalize"], precompute=random_elas.best_params_["precompute"])

        clf.fit(X, y)
        yhat = mu.forecast_model(clf, X)

        inverted = list()
        for i in range(len(yhat)):
            value = ut.inverse_difference(dataset, yhat[i], len(dataset) - i)
            inverted.append(value)
        inverted = np.array(inverted)

        forecast = inverted[-1]
        if forecast < 0:
            forecast = mu.weighted_moving_average(dataset, 1, 3)[0]
        predictions.append(forecast)
        dataset = np.append(dataset, forecast)
    predictions = [int(i) for i in predictions]
    return predictions









