# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:38:37 2019

@author: INE4KOR
"""

#%%
#Import Libraries
import pandas as pd
import numpy as np
#%%
def forecast_for_NPI(self_cluster,sku_data,plant_cluster,datasets):
    df=pd.DataFrame(columns=['sku','sales'])
    for sku in datasets:
        sales=datasets[sku].T['sales']
        index=np.where(pd.isnull(sales)==False)[0]
        sales=sales[index[0]:]
        sales=[0 if pd.isnull(i) else int(i) for i in sales]
        df=df.append({'sku':sku,'sales':sales},ignore_index=True)

    nonzero_count=len(sku_data)-(np.where(sku_data!=0)[0][0])
    same_cluster_skus=plant_cluster.groupby('pc')
    skus_list=same_cluster_skus.get_group(0).sku
    data=[df.values() for key in skus_list]
#



