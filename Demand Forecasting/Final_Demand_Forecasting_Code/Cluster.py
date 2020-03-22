# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:28:49 2019

@author: INE4KOR
"""

#%%
import pandas as pd
import numpy as np
import collections
from kmodes.kmodes import KModes
#%%
def clustering(data):
    data_cluster=dict()
    for i in range(len(data)):
        data_sku=data.iloc[i]
#        if data_sku.NPI=='NPI':
#            data_cluster[str(data_sku.key)]=0
        if data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=1
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=2
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=3
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=4
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=5
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=6
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=7
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=8
        elif data_sku.Seasonal=="Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=9

        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=10
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=11
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=12
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=13
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=14
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=15
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=16
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=17
        elif data_sku.Seasonal=="Non-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=18

        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=19
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=20
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Growing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=21
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=22
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=7
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Degrowing" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=24
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="High":
            data_cluster[str(data_sku.key)]=25
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Less":
            data_cluster[str(data_sku.key)]=26
        elif data_sku.Seasonal=="Partial-Seasonal" and data_sku.Trend=="Normal" and data_sku.spar=="Medium":
            data_cluster[str(data_sku.key)]=27

#    data_cluster = collections.OrderedDict(sorted(data_cluster.items()))
    return data_cluster
#%%
def clustering_plant(sku_list,plant_id,market_id):
    df=pd.DataFrame({'Market':market_id,'Plant':plant_id})
    km = KModes(n_clusters=4)
    pred=km.fit_predict(df)
    plant_cluster=pd.DataFrame({'sku':sku_list,'pc':pred})

    return plant_cluster




