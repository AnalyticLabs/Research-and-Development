# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:10:36 2019

@author: INE4KOR
"""
#%%
import pandas as pd

import json
#%%
def write_csv_to_json():
    path = r'C:\Users\INE4KOR\Desktop\Datasets\FDandFE\FE_A_June.csv'
    data = pd.read_csv(path)
    time = data.columns.values[6:]
#    sku = data[data.columns.values[0]]
    sku = data.pop(data.columns.values[0])
    market = data.pop(data.columns.values[0])
    plant = data.pop(data.columns.values[0])
    price = data.pop(data.columns.values[0])
    spn = data.pop(data.columns.values[0])
    abc = data.pop(data.columns.values[0])


    output = []

    for i in range(len(sku)):
        sku_data = dict()
        sku_data['sku'] = sku[i]
        sku_data['market']=market[i]
        sku_data['plant']=str(plant[i])
        sku_data['price']=float(price[i])
        sku_data['spn']=spn[i]
        sku_data['abc']=abc[i]
        sku_data['time'] = time.tolist()
        sku_data['sales'] = data.values[i].tolist()

        output.append(sku_data)
    return data, output

data, output = write_csv_to_json()



#%%
with open(r'C:\Users\INE4KOR\Desktop\Datasets\FDandFE\FE_A_June.json', 'w', encoding = 'utf-8') as f:
    json.dump(output, f, ensure_ascii = False, indent = 4)