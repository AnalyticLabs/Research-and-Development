# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:42:06 2019

@author: AKE9KOR
"""

#%%
#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

#%%
'''
'''
def dickeyfullertest(series):
    dftest = adfuller(series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index= ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    if dfoutput['p-value'] > 0.05:
        return 0 #not stationary

    else:
        return 1 #stationary
#%%
