# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:01:40 2019

@author: AKE9KOR
"""

#%%
# Import libraries
import pandas as pd
import json
import copy

#%%
def user_input():
    forecast_file = input("Enter forecast results:") 
    actuals_file = input("Enter dataset:") 
    
    forecast = retrieve_forecast(forecast_file)
    actuals = retrieve_actuals(actuals_file)
    
    return forecast, actuals

#%%
def retrieve_forecast(forecast_file):
    datasets = dict()
    
    with open(forecast_file) as json_file:
        raw_data = json.load(json_file)

        for element in raw_data:
            sku = element['sku']
            interval = element['interval']
            last_date = element['last_date']
            forecast = element['forecast']
            forecast_period = element['forecast_period']
            
            date_range = pd.date_range(last_date, periods = forecast_period + 1, freq = interval).tolist()
            date_range.pop(0) #same month as last_date, FUSO
            date_range = pd.to_datetime(date_range, infer_datetime_format = True)
            
            df_name = sku
            data = pd.DataFrame()
            data['time'] = date_range
            data['forecast'] = forecast
            
            data = data.T
            datasets[df_name] = copy.deepcopy(data)
            
    return datasets   

#%%
def retrieve_actuals(actuals_file):
    datasets = dict()
    with open(actuals_file) as json_file:
        raw_data = json.load(json_file)
        
        for element in raw_data:
            df_name = element['sku']
            data = pd.DataFrame()
            data['time'] = element['time']
            data['sales'] = element['sales']
            data['time'] = data['time'].astype(str)
            
            #FUSO
            year = data.time.str[:4]
            month = data.time.str[4:6]
            data['time'] = '28/' + month + '/' + year
            data['time'] = pd.to_datetime(data['time'], format = '%d/%m/%y', infer_datetime_format = True).dt.date
            
            data = data.T
            datasets[df_name] = copy.deepcopy(data)
            
    return datasets     


#%%
def compare(forecast_results, actual_data):
    variability = pd.DataFrame()
    error_percentage = pd.DataFrame()
    for sku in forecast_results:
        compare_df = pd.DataFrame()
        if sku in actual_data:
            forecast = forecast_results[sku].T
            forecast.columns = ['time', 'forecast']
            forecast['time'] = pd.to_datetime(forecast['time'], format = '%d/%m/%y', infer_datetime_format = True)
            forecast['time'] = forecast['time'].dt.to_period('M')
            
            forecast['sku'] = sku
            
            data = actual_data[sku].T
            data.columns = ['time', 'sales']
            data['time'] = pd.to_datetime(data['time'], format = '%d/%m/%y', infer_datetime_format = True)
            data['time'] = data['time'].dt.to_period('M')
            
            for i in range(forecast.shape[0]):
                compare_df = pd.merge(forecast, data, how = 'inner', on = [ 'time'])
            
            compare_df['diff'] = compare_df.sales - compare_df.forecast
#            compare_df['error'] = compare_df['diff'] * 100 / compare_df['forecast']
            
            variability = variability.append(compare_df[['sku', 'time', 'diff']])
#            error_percentage = error_percentage.append(compare_df[['sku', 'time', 'error']])
            
    variability.reset_index(drop = True, inplace = True)
    variability = variability.pivot(values = 'diff', index = 'sku', columns = 'time')
    variability.columns = variability.columns.strftime('%Y-%m')
    variability.index = variability.index.tolist()
    
#    error_percentage.reset_index(drop = True, inplace = True)
#    error_percentage = error_percentage.pivot(values = 'diff', index = 'sku', columns = 'time')
    
    
#    print(error_percentage)
    return variability, error_percentage

#%%
forecast, actuals = user_input()
variability, error = compare(forecast, actuals)

print(variability)



        
    