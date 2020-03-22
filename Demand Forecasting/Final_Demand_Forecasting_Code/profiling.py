# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:38:21 2019

@author: INE4KOR
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:25:14 2019

@author: INE4KOR
"""

#%%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pymannkendall as mk
from scipy import signal
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import stldecompose
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
#%%
def matur(data_sku):
  i=3
  a_ratio=3
  while a_ratio==3 and i<(len(data_sku)-3):
    data_sku_ef=data_sku[0:i]
    data_sku_el=data_sku[i:len(data_sku)]
    avg_f=np.mean(data_sku_ef)
    avg_l=np.mean(data_sku_el)
    if avg_l>5*avg_f and data_sku[i]<(1.4*avg_l):
        a_ratio=3
        i=i+1
    else:
        a_ratio=1
  starting=i-1
  return starting
#%%
def maturate(data_sku,st):
  data_sku=data_sku.reset_index(drop = True)
  a=st
  i=6
  a_ratio=3
  while a_ratio==3 and i<=(len(data_sku)-1):

    data_sku_ef1=data_sku[(i-4):i]
    data_sku_el1=data_sku[(i-3):(i+0)]
    avg_f1=np.mean(data_sku_ef1)
    avg_l1=np.mean(data_sku_el1)
    dif=abs(avg_f1-avg_l1)
    avg=(avg_f1+avg_l1)/2
    if dif<=(0.8*avg):
      a_ratio=3
      i=i+1
    else:
      a_ratio=1
  ll=[0]*3
  if 6<i and i<(len(data_sku)-4):
    ll[0]=abs(data_sku[(i+1)]-data_sku[(i+0)])
    ll[1]=abs(data_sku[(i+2)]-data_sku[(i+1)])
    ll[2]=abs(data_sku[(i+2)]-data_sku[(i+2)])
    average=(data_sku[i+0]+data_sku[i+1]+data_sku[i+2]+data_sku[i+3])/4
    ll=np.array(ll)
    if average>0:
      if len(ll[ll>average]>=1):
          starting1=a
          starting2=a+i
      else:
          starting1=a+i
          starting2=a+i
    else:
        starting1=a
        starting2=a
  else:
      starting1=a
      starting2=a
  starti=[starting1,starting2]

  return starti
#%%
def Trend_1(data_sku,star1):
    starting=star1
    data_sku3=data_sku[starting:len(data_sku)]
    data_sku3=data_sku3.reset_index(drop = True)
    index = data_sku3.ne(0).idxmax()
    data_sku4 = data_sku3[index:]       #first non zero element
    data_sku4=data_sku4.reset_index(drop = True)
    Zero=np.where((data_sku4==0)==True)
    sparsity=len(Zero)/len(data_sku4)
    if len(data_sku3)>=8 and sparsity<(0.2):
        dd=mk.original_test(data_sku3)
        if dd[0]=='increasing':
            Type_0='Growing'
        elif dd[0]=='decreasing':
            Type_0='Degrowing'
        else:
            Type_0='Normal'
    elif len(data_sku3)>=12 and sparsity>(0.2):
        #rolling-->can be applied to series
        roll_sum=pd.Series(data_sku).rolling(6).apply(np.mean)
        roll_diff=np.diff(roll_sum)
        g_1=len(np.where((roll_diff>0)==True)[0])
        l_1=len(np.where((roll_diff<0)==True)[0])
        g_p=g_1/len(roll_diff)
        l_p=l_1/len(roll_diff)
        if g_p>=0.75:
            Type_0='Growing'
        elif l_p>=0.75:
            Type_0='Degrowing'
        else:
            Type_0='Normal'

    else:
        Type_0='Normal'

  # v=c(Type_0,starting)
    return(Type_0)
#%%
def Ses_test(Yt,columns):
    seasonal=[False,False,False,False]
#    index = Yt.ne(0).idxmax()
#    Yt = Yt[index:]
#    columns=columns[index:]
    Yt=Yt.reset_index(drop=True)
    Yt,Ses_3=Out_function_1(Yt)
    Yt[np.where((Yt==0)==True)[0]]=1
    #Test 1: Statistical test of seasonal Index
    if len(Yt)>5:
        try:
            result=seasonal_decompose(Yt,model='multiplicative',freq=5)
        except:
            result=seasonal_decompose(Yt,model='additive',freq=5)
        seasonal_index=result.seasonal
        f,p=stats.ttest_1samp(seasonal_index,1)
        if p < 0.05:
            seasonal[0] = True
    #Test 2: Frequency and acf
    res = acf(Yt)
    ub=1.96/np.sqrt(len(Yt))
    for i in range(1, len(res)-1):
        if(res[i] > ub and res[i + 1] < ub):
            p = i
            if (p > 12):
                p = 12
            break
    else:
        p = 12
    d={'date':columns,'data':Yt}
    ts_data=pd.DataFrame(d)
    ts_data.set_index('date', inplace = True)

    # Test 1: periodogram
    # estimate spectral density

    freq=[0]*len(Yt)
    freq[0]=12/len(Yt)
    for i in range(1,len(Yt)):
        freq[i]=freq[i-1]+freq[0]
    freq=np.array(freq)

    f,spec = signal.periodogram(Yt)
    freq=freq[:len(spec)]
    freq_max=max(freq)
    ind=np.where(freq == freq.max())
    if freq_max < p+1.5 and freq_max > p-1.5:
        seasonal[1] = True
    #    print(freq)

    #f,spec=scipy.signal.welch(Yt,fs=100,scaling='density')
    #f,spec=signal.periodogram(Yt,nfft=None,return_onesided = True,scaling = "density",detrend='constant')
    # select higher frequencies
#    bool= (freq > 0.5 )
#    spec = spec[bool]
#    freq = freq[bool]
#    id=np.where(spec == spec.max())[0]
#    if len(id)>1:
#        id=id[0]
#    freq = freq[id]
#    if freq > 0.85 and freq < 1.15:
#        seasonal[1] = True
    # Test 2: auto-correlation function
    try:
        Tt=stldecompose.decompose(ts_data).trend #extract the trend element
    except:
        Tt=[None]*len(Yt)
    Tt=Tt.reset_index(drop=True)['data']
    if sum(Tt.isnull())==0:
        At = Yt - Tt	# detrend time series
        acf_val = acf(At)
        lag_val=[0]*75
        lag_val[0]=0
        for i in range(1,len(lag_val)):
            lag_val[i]=lag_val[i-1]+(1/12)
        ind=np.where(acf_val==np.min(acf_val))[0]
        lag=lag_val[ind[0]]
        if lag < p+1.5 and lag < p-1.5:
            seasonal[2] = True
        else:
            seasonal[2] = False
    # Test 3: seasonal model
    #seasonal ---> cycle()
    seas=[0]*len(Yt)
    j=1
    for i in range(len(Yt)):
        seas[i]=j
        j+=1
        if j>12:
            j=1
    #trend --> time()
    trend=[0]*len(Yt)
    trend[0]=1
    for i in range(1,len(Yt)):
        trend[i]=trend[i-1]+(1/12)

    d={'Yt':Yt,'seas':seas,'trend':trend}
    df = pd.DataFrame(d)
    X=df[["seas","trend"]]
    y=df["Yt"]
    m1=sm.OLS(y, X).fit()
    X1=df[["trend"]]
    m2=sm.OLS(y, X1).fit()
    bic = [m1.bic,m2.bic]
    arrind=np.where(bic==np.min(bic))[0][0]
    bic_min = bic[arrind]
    if arrind == 0:
        seasonal[3] = True
    return seasonal
#%%
def ut(x,alpha):
   m = np.median(x)
   return m + alpha *np.median(abs(x-m))
#%%
def Out_function(data_sku):
    Ses_1=[False]
    if len(data_sku[data_sku>0]>0):
        data_sku[data_sku.isnull()]=0 #replace null as zero
        ddd=data_sku
        f=data_sku
        f=f.astype(int)
        rr=np.where(f>0)[0] #index of non-zero elements
        rr_1=np.where(f==0)[0] #index of zero elements
        index =ddd.ne(0).idxmax()
        ddd=ddd[index:] #from first non-zero element
        Z_p=len(data_sku[data_sku==0])/len(data_sku) #sparsity percentage
        if len(rr_1)!=0 :
            f=f[rr[0]:]
        else:
            f=f
        f=list(f)
        ff=[0]*len(f)
        ff_1=[0]*len(f)
        fff=[0]*len(f)
        fff_1=[0]*len(f)
        #similar to bucket process
        if len(data_sku)>=24 and Z_p< 0.35:
            for i in range(0,(len(f)-12)):
                f_temp=f[i:i+6]+f[i+7:i+13]
                ff[i+6]=ut(f_temp,3)
                ff_1[i+6]=ut(f_temp,-2)
                fff[i+6]=ut(f_temp,2)
                fff_1[i+6]=ut(f_temp,-1.5)
            for i in range(0,6):
                f_temp=f[i+1:i+13]
                ff[i]=ut(f_temp,3)
                ff_1[i]=ut(f_temp,-2)
                fff[i]=ut(f_temp,2)
                fff_1[i]=ut(f_temp,-1.5)
            for i in range(0,6):
                ind=(len(f)-i-1)
                f_temp=f[ind-12:ind]
                f_temp.reverse()
                ff[len(f)-i-1]=ut(f_temp,3)
                ff_1[len(f)-i-1]=ut(f_temp,-2)
                fff[len(f)-i-1]=ut(f_temp,2)
                fff_1[len(f)-i-1]=ut(f_temp,-1.5)
            if len(rr_1)!=0:
                z1=[0]*(rr[0])+ff
                z11=[0]*(rr[0])+ff_1
                z1f=[0]*(rr[0])+fff
                z11f=[0]*(rr[0])+fff_1
            else :
                z1=ff
                z11=ff_1
                z1f=fff
                z11f=fff_1
            outliers= data_sku > z1
            outliers1=data_sku < z11
            a=np.where(outliers==True)[0]
            b=np.where(outliers1==True)[0]
            #check index where data > mad
            if len(a)>1:
                k=np.zeros([len(a)-1, len(a)])
                for i in range(0,(len(a)-1)):
                    for j in range(1,len(a)):
                        if a[j]==(a[i]+12) or a[j]==(a[i]+24): #check if the outlier is seasonal
                            k[i][j]=1
                            Ses_1[0]=True
                        else:
                            k[i][j]=0
                m=np.where(k!=0)
                z=np.unique(m).tolist()
                remove=a[z] #seasonal index
                a =np.sort(list(set(remove)^set(a)))    #ignore the seasonal index to clip the rest
            if len(b)>1:
                p=np.zeros([len(b)-1, len(b)])
                for i in range(0,len(b)-1):
                    for j in range(1,len(b)):
                        if b[j]==(b[i]+12) or b[j]==(b[i]+24):
                            p[i][j]=1
                        else:
                            p[i][j]=0

                q=np.where(p!=0)
                z_1=np.unique(q).tolist()
                remove_1=b[z_1]
                b =np.sort(list(set(remove_1)^set(b)))
            #clip data

            data_sku=np.array(data_sku)
            if len(a)>=1:
                z1f = np.array(z1f)
                data_sku[a]=z1f[a]

            if len(b)>=1:
                z11f = np.array(z11f)
                data_sku[b]=z11f[b]
        else:
            data_sku,Ses=Out_function_1(data_sku)
            Ses_1=Ses
    else:
        data_sku=data_sku
    return pd.Series(data_sku),Ses_1
#%%
#outlier using mean standard deviation
def Out_function_1(data_sku):
    Ses=[False]
    data_sku=data_sku.reset_index(drop = True)
    mean_d=np.mean(data_sku)
    sd_d=np.std(data_sku)
    O_1=mean_d+3*sd_d
    o_2=mean_d-3*sd_d
    i_1=mean_d+sd_d
    i_2=mean_d-sd_d
    a=np.where(data_sku>O_1)[0]
    b=np.where(data_sku<o_2)[0]
#    a_1=np.where(data_sku>0)[0]
    if len(a)>1:
        k=np.zeros([len(a)-1, len(a)])
        for i in range(len(a)-1):
            for j in range(1,len(a)):
                if a[j]==(a[i]+12) or a[j]==(a[i]+24):
                    k[i][j]=1
                    Ses[0]=True
                else:
                    k[i][j]=0
        m=np.where(k!=0)
        z=np.unique(m)
        remove=a[z]
        a =np.sort(list(set(remove)^set(a)))

    if len(b)>1:
        p=np.zeros([len(a)-1, len(a)])
        for i in range(len(b)-1):
          for j in range(1,len(b)):
              if b[j]==(b[i]+12) or b[j]==(b[i]+24):
                  p[i][j]=1
              else:
                  p[i][j]=0
        q=np.where(p!=0)[0]
        z_1=np.unique(q)
        remove_1=b[z_1]
        b =np.sort(list(set(remove_1)^set(b)))
    data_sku=np.array(data_sku)
    if len(a)>=1:
        data_sku[a]=i_1
    if len(b)>=1:
        data_sku[b]=i_2
    return pd.Series(data_sku),Ses

#%%
#efg = pd.read_csv('C:\\Users\\INE4KOR\\Desktop\\Datasets\\FDandFE\\FE_A_Class.csv')
#columns = list(efg.columns.values)
#sku_list = efg.pop(columns[0])
#columns.pop(0)
#columns=pd.to_datetime(columns, format='%Y/%m',infer_datetime_format=True)
def profiling_tech(datasets):
    data_prof=pd.DataFrame(columns=['key','Seasonal','Trend','NPI','spar','maturation'])
    for sku in datasets:
#        print(sku)
#        if sku!='1285715400':
#            continue
        data=datasets[sku].T
        columns=pd.to_datetime(data.index, format='%d/%m/%Y',infer_datetime_format=True)
        data=data['sales']
        data_sku=data.reset_index(drop=True)
        Type_0f=[0]*len(data_sku)
        Type_1f=[0]*len(data_sku)
        data_sku[data_sku.isnull()]=0
        data_sku = data_sku.astype(np.float32)
        if np.sum(data_sku)!=0 and len(data_sku)>=13 :
            data_sku[data_sku.isnull()]=0      #null as zero
            data_sku[data_sku<0]=0             #negative as zero
            data_sku_1=data_sku
            data_sku_1,Ses_1=Out_function(data_sku_1) #Outlier Treatment
            xx=matur(data_sku_1)                # check if there's better growth
            data_sku_10=data_sku_1[xx:len(data_sku_1)]
            dd=maturate(data_sku_10,xx)
            data_sku2=data_sku_1[dd[1]:len(data_sku_1)]
            mm=maturate(data_sku2,dd[1])
            star=xx
            if mm[0]==mm[1]:
                star=mm[0]
            Type_1=Trend_1(data_sku_1,star) #Trend func returns the type of trend
            Type=Type_1
            muturation=star
            mutation=xx
            data_sku_5=data_sku
            a_zeros=np.where((data_sku_5>0)==True)[0]
            data_sku_5=data_sku[a_zeros[0]:len(data_sku_5)]
            data_sku_5=data_sku_5.reset_index(drop = True)
            data_sku_5[np.where((data_sku_5==0)==True)[0]]=1
            if len(data_sku_5)>=24:
                Ses_2=Ses_test(data_sku,columns)
                Ses_2=np.array(Ses_2)
                ss=len(np.where(Ses_2==True)[0])
                if ss>=3 or Ses_1[0]==True:
                    Ses='Seasonal'
                elif ss>=1 and ss <=2:
                    Ses='Partial-Seasonal'
            else:
                Ses='Non-Seasonal'
            a_zeros=np.where((data_sku>0)==True)[0]
            data_sku=data_sku[a_zeros[0]:len(data_sku)]
            data_sku=data_sku.reset_index(drop = True)
            data_sku[np.where((data_sku==0)==True)[0]]=1
            if len(data_sku)<=12:
                NPI='NPI'
            else:
                NPI='Non-NPI'
            Zero=np.where((data_sku==1)==True)[0]
            percntg_zeros=len(Zero)/len(data_sku)
            if percntg_zeros<0.1:
                spar='Less'
            elif percntg_zeros>=0.1 and percntg_zeros<0.3:
              spar='Medium'
            else:
              spar='High'

        else:
            spar='High'
            Type='Normal'
            NPI='NPI'
            muturation=1
            Ses='Non-Seasonal'
        data_prof=data_prof.append({'key':sku,'Seasonal':Ses,'Trend':Type,'NPI':NPI,'spar':spar,'maturation':muturation},ignore_index=True)
    return data_prof