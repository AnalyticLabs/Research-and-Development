# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:19:55 2019

@author: AKE9KOR
"""

#%%
#Import libaries
from retrieve_data import user_input
from analytics import training
import outputs as op
#import json
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import pandas as pd
import os
os.chdir(r'C:\Users\INE4KOR\Desktop\9Sept\28Aug')

#%%
#Retrieve Data
forecast_period, datasets,details_data = user_input()


#%%
forecast_results,facc_out= training(details_data,datasets, forecast_period)

#%%
output = op.output_dataframe(forecast_results, "FD_All.csv")
#op.add_outlier_treated_data_to_csv(forecast_results, "OutlierTreatedData.csv")

#%%
#new_list = [{k: v for k, v in d.items() if k != 'dataset'} for d in forecast_results]
#new_list = [{k: v for k, v in d.items() if k != 'sku_data'} for d in new_list]
#with open(r'D:\1pls\5Assets\top_sku_FUSO_2000_results.json', 'w', encoding = 'utf-8') as f:5
#    json.dump(new_list, f, ensure_ascii = False, indent = 4, default = str)
    #%%

import matplotlib.pyplot as plt
def plot_by_years(sku, dataset, forecast, last_date, interval):
    print("SKU ", sku)
    dataset = dataset.reset_index()
    dataset.columns = ['time', 'sales']
    dataset['time'] = pd.to_datetime(dataset['time'], format = '%d/%m/%y', infer_datetime_format = True)
    dataset['time'] = dataset['time'].dt.to_period('M')

    forecast_period = len(forecast)
    date_range = pd.date_range(last_date, periods = forecast_period + 1, freq = interval).tolist()
    date_range.pop(0) #same month as last_date, FUSO
    print(type(date_range))
    date_range = pd.to_datetime(date_range, infer_datetime_format = True)
    print(type(date_range))
    year = date_range.year


    print(forecast)
    data = pd.DataFrame({'time': date_range.strftime('%Y-%m'),
                        'sales' : forecast})
    dataset.append(data)

    print(dataset.info())
    unique_years = dataset.time.year().unique()
    nyears = dataset.time.year().nunqiue()

    if(nyears > 4):
        nyears = 4
        unique_years = unique_years[-4:]

    fig, axes = plt.subplots(nrows = nyears, figsize = (12, 8))
    i = 0
    for row in axes:
        y = dataset[dataset['year'] == unique_years[i]].sales.tolist()
        x = dataset[dataset['year'] == unique_years[i]].month.tolist()
        row.plot(x, y)
        row.scatter(x, y)
        row.set_xlim(1, 12)
#        row.set_xticks(range(12))
        row.set_title("Year " + str(unique_years[i]))
        i += 1
    plt.show()
    print("--------------------")

#%%

