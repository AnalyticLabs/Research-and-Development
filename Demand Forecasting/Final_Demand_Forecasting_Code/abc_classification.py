# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:22:29 2019

@author: INE4KOR
"""
#%%
import pandas as pd
import collections
#%%
def abc_class(datasets):
    sku_total=dict()
    sku_value=dict()
    sku_class=dict()
    grand_total=0
    csum=0
    index=[]
    t=[0.6,0.25,0.15]
    for sku in datasets:
        data=datasets[sku].T
        data=data['sales']
        data = [0 if pd.isnull(i) else int(i) for i in data]

        sku_total[sku]=sum(data)
        grand_total+=sku_total[sku]

    sku_total = {k: v for k, v in sorted(sku_total.items(), reverse=True, key=lambda x: x[1])}


    for sku in sku_total:
        sku_value[sku]=sku_total[sku]/grand_total

#    print(sku_value)
    incr=i=0
    for sku in sku_value:
#        print("here")
        csum+=sku_value[sku]
        i=i+1
        if csum > t[incr]:
            csum=sku_value[sku]
            index.append(i-1)
            incr+=1
            if incr>1:
                index.append(i)
                break

#    print(index)
    for i,sku in enumerate(sku_value):
        if i <= index[0]:
            sku_class[sku]='A'
        elif i<=index[1]:
            sku_class[sku]='B'
        elif i>=index[2]:
            sku_class[sku]='C'

#    sku_class = collections.OrderedDict(sorted(sku_class.items()))
#    return list(sku_class.values())
    return sku_class