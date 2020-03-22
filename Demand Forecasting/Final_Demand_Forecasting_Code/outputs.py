# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:04:49 2019

@author: AKE9KOR
"""

#%%
#Importing Libraries
import json
import preprocess as pp
import pandas as pd


#%%
def best_model_details_ts(best_model,bias,accuracy,best_order):
    ts_details=dict()
    param=dict()
    ts_details['model']=best_model
    ts_details['bias']=bias
    ts_details['accuracy']=accuracy
    if best_model in ['AR','ARMA','ARIMA']:
        param['p']=best_order[0]
    if best_model in ['ARIMA']:
        param['d']=best_order[1]
    if best_model in ['MA','ARMA','ARIMA']:
        param['q']=best_order[2]
    ts_details['parameters']=param


    return ts_details

#%%
def best_model_details_ml(best_model,bias,accuracy,parameter):
    ml_details=dict()
    ml_details['model']=best_model
    ml_details['bias']=bias
    ml_details['accuracy']=accuracy
    ml_details['parameters']=parameter

    return ml_details


#%%
#Add predictions and RMSE for all models to json
def add_trained_models_data(sku, expected, rmse, predictions, best_model, interval, best_order):
    sku_data = dict()
    sku_data['sku'] = sku
    sku_data['expected'] = expected
    sku_data['best_model'] = best_model
    sku_data['best_order'] = best_order
    sku_data['interval'] = interval

    predictions_list = []
    for key in rmse.keys():
        model_results = dict()
        model_results['model'] = key
        model_results['rmse'] = rmse[key]
        model_results['predictions'] = predictions[key]
        predictions_list.append(model_results)

    sku_data['predictions'] = predictions_list
    return sku_data

#%%
def write_trained_data_to_json(outputs):
    outputs = json.dumps(outputs)
    with open('trained_results.json', 'w', encoding = 'utf-8') as f:
        json.dump(outputs, f, ensure_ascii = False, indent = 4)

#%%
def add_forecasted_results(sku, dataset, data, output):
    sku_data = dict()
    sku_data['sku'] = sku
#    sku_data['dataset'] = dataset
#    sku_data['sku_data'] = data

    for key in output:
        sku_data[key] = output[key]

    return sku_data


#%%
def output_dataframe(forecast_results, file_name):
    cols = ['sku', 'forecast_period',
            #'error_ml', 'error_ts', 'error_en',
            #'weight_ts', 'weight_ml',
            #'model_ml', 'model_ts',
            'interval', 'last_date', 'forecast_values']#, 'forecast_ml', 'forecast_ts', 'forecast_en'

    output = pd.DataFrame.from_dict(forecast_results)
    output = output.ix[:, cols]

    f = output.forecast_values.apply(pd.Series)
    output = pd.concat([output[:], f[:]], axis=1)
    output.drop('forecast_values', axis = 1, inplace = True)
#
#    f = output.forecast_ml.apply(pd.Series)
#    f = f.rename(columns = lambda x: 'ml + '+ str(int(x) + 1))
#    output = pd.concat([output[:], f[:]], axis=1)
#    output.drop('forecast_ml', axis = 1, inplace = True)
#
#    f = output.forecast_ts.apply(pd.Series)
#    f = f.rename(columns = lambda x: 'ts + '+ str(int(x) + 1))
#    output = pd.concat([output[:], f[:]], axis=1)
#    output.drop('forecast_ts', axis = 1, inplace = True)
#
#    f = output.forecast_en.apply(pd.Series)
#    f = f.rename(columns = lambda x: 'en + '+ str(int(x) + 1))
#    output = pd.concat([output[:], f[:]], axis=1)
#    output.drop('forecast_en', axis = 1, inplace = True)

    output.to_csv(file_name)

    return output

#%%
def add_outlier_treated_data_to_csv(forecast_results, file_name):
    data = pd.DataFrame()
    for f in forecast_results:
        sku = f['sku']
        data[sku] = f['sku_data'].T.squeeze()

    data = data.T
    data.to_csv(file_name)