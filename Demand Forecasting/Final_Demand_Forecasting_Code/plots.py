# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:26:52 2019

@author: AKE9KOR
"""

#%%
#Importing libraries
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import util as ut

#%%
def acf_plot1(dataset,freq):
#    plot_acf(dataset)
    res = acf(dataset)
    ub=1.96/np.sqrt(len(dataset))
    for i in range(1, len(res)-1):
        if(res[i] > ub and res[i + 1] < ub):
            p = i
            if (p > freq):
                p = freq
            break
    else:
        p = freq
    return p
#%%
def acf_plot(dataset,freq):
    res = acf(dataset)
    plot_acf(dataset)
    acfval=[0]*len(res)
    ub=1.96/np.sqrt(len(dataset))
    p1=1
    for i in range(len(res)):
        acfval[i]=abs(res[i])
    acfval.sort(reverse=True)
    acfval=np.array(acfval[1:])
    pshort=np.array(acfval[0:3])
    pshortind=[0]*len(pshort)
#    print(pshort)
    for i in range(len(pshort)):
        pshortind[i]=np.where(abs(res)==pshort[i])[0][0]
    ind=np.where(acfval>ub)[0]
    finalacf=acfval[ind]
    plist=[0]*len(finalacf)
    for i in range(len(finalacf)):
        plist[i]=np.where(abs(res)==finalacf[i])[0][0]
#    return pshortind,plist
#    print(plist)
    while len(finalacf)>0:
        p1=np.where(abs(res)==max(finalacf))[0][0]
        if p1 > freq:
            finalacf=finalacf[1:]
        else:
            break
    return p1,pshortind,plist




#%%
def pacf_plot1(dataset,freq):
#    plot_pacf(dataset)
    res = pacf(dataset)
    ub=1.96/np.sqrt(len(dataset))
    for i in range(1, len(res)-1):
        if(res[i] > ub and res[i + 1] < ub):
            q = i
            if (q > freq/2):
                q = freq/2
            break
    else:
        q = freq/2
    return int(q)
#%%
def pacf_plot(dataset,freq):
    res = pacf(dataset)
    plot_pacf(dataset)
    pacfval=[0]*len(res)
    ub=1.96/np.sqrt(len(dataset))
    q1=0
    for i in range(len(res)):
        pacfval[i]=abs(res[i])
    pacfval.sort(reverse=True)
    pacfval=np.array(pacfval[1:])
    ind=np.where(pacfval>ub)[0]
    finalpacf=pacfval[ind]
    while len(finalpacf)>0:
        q1=np.where(abs(res)==max(finalpacf))[0]
        q1=q1[0]
        if q1 > int(freq/2):
            finalpacf=finalpacf[1:]
        else:
            break
    return q1

#%%

