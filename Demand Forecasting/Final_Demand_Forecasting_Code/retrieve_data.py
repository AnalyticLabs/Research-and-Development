# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:12:54 2019

@author: AKE9KOR
"""

#%%
#Import libaries
import pandas as pd
import numpy as np
import json
import copy
import os

#%%
def user_input():
    forecast_period = input("What is the Future Forecast period? ")
    forecast_period = int(forecast_period)

    input_data = input("Enter dataset: ")

    file_type = os.path.splitext(input_data)[1]
    if file_type == '.csv':
        dataset = pd.read_csv(input_data)

    elif file_type == '.json':
        dataset,detailed_data = retrieve_data(input_data)




    return forecast_period, dataset, detailed_data
#%%
def retrieve_data(path):
    datasets = dict()
    detailed_data=pd.DataFrame(columns=['sku','market','plant','price','spn','abc_data'])
    with open(path) as json_file:
        raw_data = json.load(json_file)

        for element in raw_data:
            df_name = element['sku']
            data = pd.DataFrame()
            data['time'] = element['time']
            data['sales'] = element['sales']
#            print(data['sales'])
#            data['time'] = data['time'].astype(str)
            data['sales'] = [np.nan if pd.isnull(i) else int(i) for i in data['sales']]

#            FUSO
            year = data.time.str[:4]
            month = data.time.str[4:6]
            data['time'] = '28/' + month + '/' + year
            data['time'] = pd.to_datetime(data['time'], format = '%d/%m/%y', infer_datetime_format = True).dt.date

            data = data.T
            data = data.rename(columns = data.iloc[0]).drop(data.index[0])
            datasets[df_name] = copy.deepcopy(data)
            detailed_data=detailed_data.append({'sku':str(element['sku']),'market':element['market'],'plant':element['plant'],'price':element['price'],'spn':element['spn'],'abc_data':element['abc']},ignore_index=True)


    return datasets,detailed_data
#%%
#def retrieve_details(path):
#    detailed_data=pd.DataFrame(columns=['sku','market','plant','price','spn','abc_data'])
#    with open(path) as json_file:
#        details_data = json.load(json_file)
#
#        for element in details_data:
#            detailed_data=detailed_data.append({'sku':str(element['sku']),'market':element['market'],'plant':element['plant'],'price':element['price'],'spn':element['spn'],'abc_data':element['abc']},ignore_index=True)
#
#    return detailed_data

