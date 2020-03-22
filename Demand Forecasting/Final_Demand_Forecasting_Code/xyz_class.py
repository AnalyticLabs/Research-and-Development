# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:17:52 2019

@author: INE4KOR
"""
#%%
import pandas as pd
import collections
#%%
def xyz_class(sku_price,total_price):
    sku_xyz_class=dict()
    sku_price_val=dict()
    for sku in sku_price:
        sku_price_val[sku]=float(format(sku_price[sku]/total_price,'.3f'))

    sku_price_val = {k: v for k, v in sorted(sku_price_val.items(), reverse=True, key=lambda x: x[1])}

    incr=i=0
    csum=0
    index=[]
    t=[0.6,0.25,0.15]
    for sku in sku_price_val:
        csum+=sku_price_val[sku]
        i+=1
        if csum > t[incr]:
            csum=sku_price_val[sku]
            incr=incr+1
            index.append(i-1)
            if incr>1:
                index.append(i)
                break

    for i,sku in enumerate(sku_price_val):
        if i <= index[0]:
            sku_xyz_class[sku]='X'
        elif i<=index[1]:
            sku_xyz_class[sku]='Y'
        elif i>=index[2]:
            sku_xyz_class[sku]='Z'

#    sku_xyz_class = collections.OrderedDict(sorted(sku_xyz_class.items()))
#    return list(sku_xyz_class.values())
    return sku_xyz_class
#%%
