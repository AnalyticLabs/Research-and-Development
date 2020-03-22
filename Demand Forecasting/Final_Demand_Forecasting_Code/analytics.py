# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:23:42 2019

@author: AKE9KOR
"""

#%%
#Import libraries
import pandas as pd
import numpy as np
import math
import train
import util as ut
import tests
import plots
import preprocess as pp
import forecast as ft
import outputs as op
import modeling_util as mu
import model_config as mc
import abc_classification as abc
import xyz_class as xyz
import copy
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import ts_algo as ts
import newproduct as npm
import profiling
import Cluster
#%%

'''
Train the models for each SKU and forecast the predictions

'''
def training(details_data,datasets, forecast_period):

    facc_out=dict()
    rsq=dict()
    price= details_data['price']
    price=[float(i) for i in price]
    price=pd.Series(price).fillna(0).tolist()

    sku_list= details_data['sku']
    market=details_data['market']
    plant= details_data['plant']
    spn= details_data['spn']
    abc_data= details_data['abc_data']


    #Profiling
    data_prof=profiling.profiling_tech(datasets)

    #Clustering based on nature
    data_cluster=Cluster.clustering(data_prof)

    total_price=np.sum(price)
    sku_price=dict()
    for i,sku in enumerate(sku_list):
        sku_price[str(sku)]=price[i]

    #Market Based Clustering
#    plant_cluster=Cluster.clustering_plant(sku_list,plant,market)

    #XYZ based on unit cost
    xyz_data=xyz.xyz_class(sku_price,total_price)
    #ABC based on volume
    abc_alter=abc.abc_class(datasets)

#    trained_outputs = []
    forecast_results = []
    num = 0
    for incr,sku in enumerate(datasets):
        num += 1
#        if sku!='1702460700':
##        if num!=1:
#            continue
        prof=data_prof.iloc[incr]
        print("------------------------------------------------------------")
        print("Running SKU %d: %s..." % (num, sku))
        print("cluster :  ",data_cluster[sku])

        raw_data = copy.deepcopy(datasets[sku].T)
        output = ut.init_output(forecast_period, raw_data,prof)
        output['unit_cost']=float(price[incr])
        output['market']=str(market[incr])
        output['plant']=str(plant[incr])
        if pd.isnull(abc_data[incr])==True:
            output['Variability_Segment']=abc_alter[sku]
        else:
            output['Variability_Segment']=abc_data[incr]
        output['Velocity_segment']=xyz_data[sku]
        output['spn']=spn[incr]




        dataset = raw_data.copy()

        dataset = dataset[:-1]
#        dataset = pp.dateformat(dataset)
#        dataset, interval = pp.impute_missing_dates(dataset)
#        print(interval.days)

        if((dataset['sales'] == 0).all() == True or (set([math.isnan(x) for x in dataset['sales']]) == {True})):
#            print(dataset['sales'])
            print("All zeros/NaNs")
            forecast = [0] * forecast_period
            output['forecast_values'] = ut.assign_dates(forecast, 'forecast', dataset.tail(1))
            output['facc'], output['mape'], output['bias'] = ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
            facc_out[sku]=np.mean(ft.calculate_validation_facc(forecast, forecast))
            forecast_results = ft.output_forecast(sku, dataset, datasets[sku].T, output, forecast_results)
            continue

        sku_data = dataset.astype(np.float32)
        sku_data = pp.read_from_first_sales(sku_data['sales'])


#size--->outlier bucket size
#sparse_size ---> number of zeros to categorize as sparse data
#freq ---> seasonality
        interval=30
        size=6
        sparse_size=10
        freq=12
#        size,sparse_size,freq=pp.get_bucket_size(interval)

        test_nan=pd.DataFrame(sku_data[-freq:])
        test_nan=test_nan['sales']

#if last 1 year is NaN, impute data with zero and forecast is MA(6)

        if sum(test_nan.isnull())>=freq:
            print("Last 1 year NaN")
            sku_data = pp.data_imputation_zero(test_nan)
            sku_data=sku_data[:-5]
            expected=[0]*5
            forecast = mu.moving_average(sku_data,forecast_period,6)
            output['forecast_values'] = ut.assign_dates(forecast, 'forecast', dataset.tail(1))
            output['facc'], output['mape'], output['bias'] = ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
            facc_out[sku]=np.mean(ft.calculate_validation_facc(expected, forecast))
            forecast_results = ft.output_forecast(sku, dataset, sku_data, output, forecast_results)
            continue

#if # NaNs more than 60% impute with 0 else impute with values

        if sum(pd.isnull(sku_data))>(0.6*len(sku_data)):
            print("Nan Greater than 60%")
            sku_data = pp.data_imputation_zero(sku_data)

        else:
            print("Nan less than 60%")
            sku_data = pp.data_imputation(sku_data,freq)
            sku_data=sku_data[0]

        sku_data = pp.read_from_first_sales(sku_data)

#After reading from first non-zero if data is insufficient ---> weighted MA(3)

        if len(sku_data) < 20:
            try:
                print("Weighted Moving Average")
                forecast = mu.weighted_moving_average(sku_data,forecast_period,3)
                output['forecast_values'] = ut.assign_dates(forecast, 'forecast', dataset.tail(1))
                output['facc'], output['mape'], output['bias'] = ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
                facc_out[sku]=ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
                forecast_results = ft.output_forecast(sku, dataset, sku_data, output, forecast_results)
            except:
                print("Less than 3")
                print(sku_data)
                forecast = mu.moving_average(sku_data, forecast_period, len(sku_data))
                output['forecast_values'] = ut.assign_dates(forecast, 'forecast', dataset.tail(1))
                output['facc'], output['mape'], output['bias'] = ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
                facc_out[sku]=ft.calculate_forecast_accuracy(raw_data.iloc[-1], forecast[0])
                forecast_results = ft.output_forecast(sku, dataset, sku_data, output, forecast_results)

            continue

        data_copy=sku_data.copy()
        data_copy=np.array(data_copy)
#        plt.figure()
#        plt.plot(data_copy)


        index1,index2,sflag1,sflag2=pp.Sesonal_detection(sku_data)
        sku_data = pp.outlier_treatment_tech(sku_data,interval,size)
        sku_data=np.array(sku_data[0])


        if sflag1==1:
            sku_data[index1]=data_copy[index1]
        if sflag2==1:
            sku_data[index2]=data_copy[index2]
        else:
            sku_data=sku_data
#        plt.plot(sku_data)
#        plt.show()
#        continue
        sku_data=pd.DataFrame(sku_data)

        #Testing Stationarity
        d = 0
        df_test_result = tests.dickeyfullertest(sku_data.T.squeeze()) #pd.Series(sku_data[0])

        while df_test_result == 0:
            d += 1
            if d == 1:
                new_data = ut.difference(sku_data[0].tolist())
            else:
                new_data = ut.difference(new_data)
            df_test_result = tests.dickeyfullertest(new_data)

        sample=np.array(sku_data)
        repeat=mu.check_repetition(sample, freq , 1, len(sample))
        #Finding p and q value
        try:
            if d == 0:
                p1,ps,pl = plots.acf_plot(sku_data,freq)
                q = plots.pacf_plot(sku_data,freq)
                data = sku_data
            else:

                p,ps,pl = plots.acf_plot(new_data,freq)
                q = plots.pacf_plot(new_data,freq)
                data = new_data

            if repeat in ps:
                p=repeat
            elif repeat in pl:
                p=repeat
            else:
                p=pl[0]
            if p > freq:
                p=freq
        except:
            p=1
            q=1
            data=sku_data

        data=sku_data
        best_order = (p, d, q)
        print("BEST ORDER :",best_order)
        #TODO: Calculate tsize
        tsize=5
#        tsize = int(0.2*len(data))
#        print(test)
        expected = data[-tsize:].reset_index(drop = True)
        expected = [float(i) for i in expected.values]
#        print("Dimension: ", data.shape)
        train_6wa=sku_data[0:-tsize]
        predictions_ML,rmse_ML = train.time_series_using_ml(sku_data, tsize, best_order,data_cluster[sku])
        rmse_ARIMA, rmse_ES, rmse_naive,rmse_ma, predictions_ARIMA, predictions_ES , predictions_naive, predictions_ma = train.time_series_models(freq,sku_data, data, tsize, best_order,data_cluster[sku])
        print("Modeling done")

        rmse_TS = rmse_ARIMA.copy()
        rmse_TS.update(rmse_ES)
        rmse_TS.update(rmse_naive)
        rmse_TS.update(rmse_ma)

        predictions = predictions_ML
        predictions.update(predictions_ARIMA)
        predictions.update(predictions_ES)
        predictions.update(predictions_naive)
        predictions.update(predictions_ma)

        if data_cluster[sku] in [1,4,7,10,13,16,19,22,25]:
            rmse_Croston,predictions_Croston=mu.Croston_TSB(sku_data,tsize)
            rmse_TS.update(rmse_Croston)
            predictions.update(predictions_Croston)


        rmse_vol_ml=dict()
        for key in rmse_ML:
            std = np.std(rmse_ML[key])
            mean = np.mean(rmse_ML[key])
            rmse_vol_ml[key]= mean
#            if std == 0:
#                rmse_vol_ml[key]= mean
#            else:
#                rmse_vol_ml[key] = mean/std

        rmse_vol_ts=dict()
        for key in rmse_TS:
            mean = np.mean(rmse_TS[key])
            std = np.std(rmse_TS[key])
            rmse_vol_ts[key] = mean
#            if std == 0:
#                rmse_vol_ts[key] = mean
#            else:
#                rmse_vol_ts[key]= mean/std


        #Top 3 models
        best_models_ml =  sorted(rmse_vol_ml, key=rmse_vol_ml.get, reverse=False)[:3]
        best_models_ts =  sorted(rmse_vol_ts, key=rmse_vol_ts.get, reverse=False)[:3]

#        forecasts_ml = dict()
#        validation_ml = dict()
        bias_ml = []
        accuracy_ml = []
        for model in best_models_ml:
#            temp = ft.model_predict(model, best_order,data, forecast_period)
#            forecasts_ml[model] = [0 if i < 0 else int(i) for i in temp]
#            validation_ml[model] = predictions[model]
            bias_ml.append((sum(expected) - sum(predictions[model]))/len(expected))
            accuracy_ml.append(mu.calculate_facc(expected, predictions[model]))
        bias_ml=[float(format(i,'.3f')) for i in bias_ml]
        accuracy_ml=[float(format(i,'.3f')) for i in accuracy_ml]


#        forecasts_ts = dict()
#        validation_ts = dict()
        bias_ts = []
        accuracy_ts = []
        for model in best_models_ts:
#            temp = ft.model_predict(model, best_order, sku_data, forecast_period,repeat)
#            forecasts_ts[model] = [0 if i < 0 else int(i) for i in temp]
#            validation_ts[model] = predictions[model]
            bias_ts.append((sum(expected) - sum(predictions[model]))/len(expected))
            accuracy_ts.append(mu.calculate_facc(expected, predictions[model]))
        bias_ts=[float(format(i,'.3f')) for i in bias_ts]
        accuracy_ts=[float(format(i,'.3f')) for i in accuracy_ts]

        #For one ensemble
        error_ml = min(rmse_vol_ml.values())
        error_ts = min(rmse_vol_ts.values())


        best_models = [min(rmse_vol_ml, key = lambda x : rmse_vol_ml.get(x)), min(rmse_vol_ts, key = lambda x : rmse_vol_ts.get(x))]
        print("BEST MODELS :",best_models)
        print("ERRORS OF BEST MODELS :",error_ml, error_ts)
        forecast_ml,param_val_fore = ft.model_predict(best_models[0], best_order,data, forecast_period)

        if best_models[1]=='Croston':
            rmse_Croston,forecast_ts=mu.Croston_TSB(sku_data,forecast_period)
            forecast_ts=forecast_ts['Croston']
        else:
            forecast_ts,param_val = ft.model_predict(best_models[1], best_order, sku_data, forecast_period,repeat)

        forecast_ml = [0 if i < 0 else int(i) for i in forecast_ml]
        forecast_ts = [0 if i < 0 else int(i) for i in forecast_ts]

        weight_ts,weight_ml=ut.weight_calculation(data,best_models,best_order)
        print("weight ts:" ,weight_ts)
        print("weight ml:" ,weight_ml)

        Vm=predictions[best_models[0]]
        Vt=predictions[best_models[1]]

        Ve=ut.method_ensemble(Vm,Vt,weight_ml,weight_ts,tsize)
        error_en=mu.calculate_rmse('Ensemble', expected, Ve)

        bias_en=[]
        accuracy_en=[]

        bias_en.append((sum(expected) - sum(Ve))/len(expected))
        accuracy_en.append(mu.calculate_facc(expected, Ve))
        bias_en=[float(format(i,'.3f')) for i in bias_en]
        accuracy_en=[float(format(i,'.3f')) for i in accuracy_en]
        #Ensemble of six month naive and weighted average
        V6wa,rmse_6wa = ts.model_Naive('naive6wa',train_6wa, tsize, (0,0,0), 0 , train_flag = 1)
        error_6wa=np.mean(rmse_6wa)
        forecast_6wa,param_val= ft.model_predict('naive6wa', best_order,data, forecast_period)

        forecast_en=ut.method_ensemble(forecast_ml,forecast_ts,weight_ml,weight_ts,forecast_period)

        output['forecast_period'] = forecast_period
        output['interval'] = 'M'
        output['best_models_ml'] = best_models_ml
        output['best_models_ts'] = best_models_ts
        output['bias_ml'] = bias_ml
        output['bias_ts'] = bias_ts
        output['bias_en'] = bias_en
        output['accuracy_ml'] = accuracy_ml
        output['accuracy_ts'] = accuracy_ts
        output['accuracy_en'] = accuracy_en
        output['TS']=op.best_model_details_ts(best_models[1],bias_ts[0],accuracy_ts[0],best_order)
        output['ML']=op.best_model_details_ml(best_models[0],bias_ml[0],accuracy_ml[0],param_val_fore)
        output['Ensemble']={"bias":bias_en[0],"accuracy":accuracy_en[0]}



        error_min_model=min(error_ml,error_ts,error_en)


        print("Errors:", )
        print("ML:", error_ml)
        print("TS:", error_ts)
        print("Ensemble:", error_en)
        print("six_naive_WA",error_6wa)



        min_error=min(error_min_model,error_6wa)

        if min_error==error_ml:
            ftt=forecast_ml
        elif min_error==error_ts:
            ftt=forecast_ts
        elif min_error==error_en:
            ftt=forecast_en
        else:
            ftt=[]


        if min_error==error_6wa or all(elem == ftt[0] for elem in ftt)==True:
            print("Best forecast from six naive")
            forecast = forecast_6wa
            output['validation'] = ut.assign_dates(V6wa, 'validation', dataset.tail(5))
            validation_facc = ft.calculate_validation_facc(expected, V6wa)
            output['validation_facc'] = ut.assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_ml:
            print("Best forecast from ML")
            forecast = forecast_ml
            output['validation'] = ut.assign_dates(Vm, 'validation', dataset.tail(5))
            validation_facc = ft.calculate_validation_facc(expected, Vm)
            output['validation_facc'] = ut.assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_en:
            print("Best forecast from Ensemble")
            forecast=forecast_en
            output['validation'] = ut.assign_dates(Ve, 'validation', dataset.tail(5))
            validation_facc = ft.calculate_validation_facc(expected,Ve)
            output['validation_facc'] = ut.assign_dates(validation_facc, 'val_facc', dataset.tail(5))
        elif min_error==error_ts:
            print("Best forecast from TS")
            forecast=forecast_ts
            output['validation'] = ut.assign_dates(Vt, 'validation', dataset.tail(5))
            validation_facc = ft.calculate_validation_facc(expected, Vt)
            output['validation_facc'] = ut.assign_dates(validation_facc, 'val_facc', dataset.tail(5))
#
        print("Forecasts:")
        print("ML:", forecast_ml)
        print("TS:", forecast_ts)
        print("Ensemble:", forecast_en)
        print("Best Forecast",forecast)

        output['forecast_values'] = ut.assign_dates(forecast, 'forecast', dataset.tail(1))
        output['facc'], output['mape'], output['bias']= ft.calculate_forecast_accuracy(raw_data.iloc[-1].sales, forecast[0])
        facc_out[sku]=np.mean(validation_facc)

        output['forecast_ml'] = ut.assign_dates(forecast_ml, 'forecast', dataset.tail(1))
        output['forecast_ts'] = ut.assign_dates(forecast_ts, 'forecast', dataset.tail(1))
        output['forecast_en'] = ut.assign_dates(forecast_en, 'forecast', dataset.tail(1))
        output['model_ml'] = best_models[0]
        output['model_ts'] = best_models[1]


        forecast_results = ft.output_forecast(sku, dataset, sku_data, output, forecast_results)


        ft.plot_all_forecasts(dataset, sku_data, forecast,forecast_en,forecast_ml, forecast_ts, sku)

    return forecast_results,facc_out