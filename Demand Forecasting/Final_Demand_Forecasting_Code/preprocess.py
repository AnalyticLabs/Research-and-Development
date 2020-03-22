# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:11:45 2019

@author: AKE9KOR
"""

#%%
#Import libraries
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import statistics
#%%
def mean_standard_deviation(dataset):
    mu = np.mean(dataset.values)
    sd = np.std(dataset.values)

    ub = mu + (3 * sd)
    lb = mu - (3* sd)

    return lb,ub

#%%
#TODO: Send alpha, beta
def median_absolute_deviation(dataset, median):
    median_list = list()
    dataset.reset_index(drop = True, inplace = True)
    for i in range(0, len(dataset)):
        value = dataset.T[i] - median
        median_list.append(value)
    ms = np.abs(median_list)
    mad = np.median(ms)
    ub = median + (3 * mad)
    lb = median - (3 * mad)


    return ub,lb
#%%
def dateformat(dataset):
    dataset['time'] = dataset.index.tolist()
    dataset['time']=pd.to_datetime(dataset['time'], format='%m/%d/%y',infer_datetime_format=True)
    dataset.set_index('time', inplace = True)
    return dataset

#%%
def outlier_treatment(dataset):
    median = np.median(dataset)
    if median == 0:
        ub,lb = mean_standard_deviation(dataset)
    else:
        ub,lb = median_absolute_deviation(dataset, median)
    new_dataset = np.clip(dataset, lb, ub)
    return new_dataset
#%%
#pt--->outlier bucket size
#sindex ---> number of zeros to categorize as sparse data
#freq ---> seasonality
def get_bucket_size(interval):
    interval_type = find_interval_type(interval) #aggregation (weekly/monthly)
    if interval_type == 'W':
        pt=12
        sindex=24
        freq=52
    elif interval_type=='M'or interval_type=='Random':
        pt=6
        sindex=10
        freq=12
    elif interval_type=='Y':
        pt=2
        sindex=0
        freq=0
    return pt,sindex,freq

#%%
def outlier_treatment_tech(dataset,interval,pt): #dataset, actual interval in numbers, bucket size
    #TODO: remove interval
    start=0
    end=pt
    sku_data=[0]*len(dataset)
    while end < len(dataset):
        sku_data[start:end]=outlier_treatment(dataset[start:end])
        start=end
        end+=pt
    if start < len(dataset):
        sku_data[start:len(dataset)]=outlier_treatment(dataset[start:end])
#        sku_data=sku_data.append(outlier_treatment(dataset[start:len(dataset)]))
    sku_data=pd.DataFrame(sku_data)
    return sku_data
#%%
def Sesonal_detection(sku_data):
    median = np.median(sku_data)

    if median == 0:
        ub,lb = mean_standard_deviation(sku_data)
    else:
        ub,lb = median_absolute_deviation(sku_data, median)
    outliers1= sku_data > ub
    outliers2 = sku_data < lb
    a=np.where(outliers1==True)[0]
    b=np.where(outliers2==True)[0]
    flag1=flag2=1
    if len(a)==0:
        flag1=0
        remove1=[]
    if len(b)==0:
        flag2=0
        remove2=[]

    if flag1==1:
        k=np.zeros([len(a)-1, len(a)])
        for i in range(0,(len(a)-1)):
            for j in range(1,len(a)):
                if a[j]==(a[i]+12) or a[j]==(a[i]+24):
                    k[i][j]=1
                else:
                    k[i][j]=0
        m=np.where(k!=0)
        z=np.unique(m).tolist()
        remove1=a[z]
    if flag2==1:
        q=np.zeros([len(b)-1, len(b)])
        for i in range(0,(len(b)-1)):
            for j in range(1,len(b)):
                if b[j]==(b[i]+12) or b[j]==(b[i]+24):
                    q[i][j]=1
                else:
                    q[i][j]=0
        n=np.where(q!=0)
        z1=np.unique(n).tolist()
        remove2=b[z1]
    return remove1,remove2,flag1,flag2

#%%
def impute_missing_dates(dataset):
    interval,start_date,end_date=find_interval(dataset.index)
    drange = pd.date_range(start_date, end_date, freq = interval)
    comp_data = pd.DataFrame(drange, columns = ['time'])
    sales=dataset['sales'].to_dict()
    comp_data['sales'] = comp_data['time'].map(sales)
    comp_data.drop('time', axis = 1, inplace = True)
    return comp_data, interval

#%%
def find_interval(date):
    date=pd.to_datetime(date, format='%m/%d/%Y',infer_datetime_format=True)
    diff=[]
    for i in range(len(date)-1):
        interval=date[i+1]-date[i]
        diff.append(interval)
    mode=statistics.mode(diff)

    return mode,date[0],date[-1]


#%%
def find_interval_type(interval):
    interval=interval.days
    if interval==7:
        itype='W'
    elif interval==30 or interval==31:
        itype='M'
    elif interval==365:
        itype='Y'
    else:
        itype='Random'

    return itype

#%%
def data_imputation_zero(dataset):
    dataset.fillna(0, inplace = True)
    return dataset

#%%
#TODO: Send as transpose of dataset
def data_imputation(dataset,freq):
    #Taking the mean of nearest neighbours to fill NA

    data_forward = dataset.fillna(method = 'ffill')
    data_back = dataset.fillna(method = 'bfill')
    data_back.fillna(0, inplace = True)
    data_forward.fillna(0, inplace = True)
    new_data = (data_forward.values + data_back.values) / 2
    dataset=pd.DataFrame(dataset.values)
    imput=dataset.isnull()
    imput=imput[0]
    dataset=dataset[0]
    for i in range(len(dataset)):
        div_factor = 3
        if imput[i]==True:

            if i - freq < 0:
                prev_value = 0
                div_factor -= 1
            else:
                prev_value = dataset[i - freq]
#Outside boundary or next value is NaN, set previous as 0
            if i + freq >= len(dataset) or imput[i + freq]==True:

                next_value = 0
                div_factor -= 1
#                print(next_value, div_factor)
            #Fetch next value
            else:
#                print("Inside")
                next_value = dataset[i + freq]
#                print(next_value)

#            print("NEW DATA", new_data[i])
            dataset[i] = (new_data[i] + prev_value + next_value)/div_factor

#            print(dataset[i])
#            print(dataset)

    df = pd.DataFrame(dataset)

#    print(df)
    return df

#%%
#reading from first non-zero
def read_from_first_sales(sku_data):
    test=pd.isnull(sku_data)
    index=np.where(test==False)[0]
    index=index[0]
    sku_data = sku_data[index:]
    sku_data=sku_data.reset_index(drop = True)
    return sku_data


#%%
def new_forecast(data,forecast,forecast_period):

    sample_data=data[-forecast_period:]
    sample_data=pd.DataFrame(sample_data)
    forecast=pd.DataFrame(forecast)
#    print(forecast)
    median_data=np.median(sample_data)
    median_fore=np.median(forecast)
#    print("median data: ",median_data)
#    print("median fore: ",median_fore)
    if median_data == 0:
        ub_d,lb_d = mean_standard_deviation(sample_data)
    else:
        ub_d,lb_d = median_absolute_deviation(sample_data, median_data)
    if median_fore == 0:
        ub_f,lb_f = mean_standard_deviation(forecast)
    else:
        ub_f,lb_f = median_absolute_deviation(forecast, median_fore)
#    print("sample", ub_d,lb_d)
#    print("forecast", ub_f,lb_f)
    if ub_f >ub_d:
        ub=ub_d
    else:
        ub=ub_f
    if lb_f >lb_d:
        lb=lb_d
    else:
        lb=lb_f
#    print("---------------------")
#    print("ub",ub)
#    print("lb",lb)
    forecast=np.clip(forecast,lb,ub)
#    print("aaaaaaaaaaaaaa",forecast)
    return forecast