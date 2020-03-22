# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:34:51 2019

@author: 91953
"""

#%% importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
import xgboost as xgb
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

#%% Functions Definitions

def rmse(y_actual, y_predicted):
    rms = sqrt(mean_squared_error(y_actual, y_predicted))
    return rms

#%% Reading The Data
df = pd.read_csv("C:\\Users\\91953\\Desktop\\Analytic Labs\\Teacheron\\Debodeep\\Stock Market\\Data\\AAPL_1990.csv")

#%% looking at the first five rows of the data

print(df.head())
print('\n Shape of the data:')
print(df.shape)

#%% setting the index as date

df['Date'] = pd.to_datetime(df.Date,format='%d-%m-%Y')
df.index = df['Date']

#%%

# Relative Strength Index
# Avg(PriceUp)/(Avg(PriceUP)+Avg(PriceDown)*100
# Where: PriceUp(t)=1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)>0};
#        PriceDown(t)=-1*(Price(t)-Price(t-1)){Price(t)- Price(t-1)<0};

def rsi(values):
    up = values[values>0].mean()
    down = -1*values[values<0].mean()
    return 100 * up / (up + down)

# Add Momentum_1D column for all 15 stocks.
# Momentum_1D = P(t) - P(t-1)

df['Momentum_1D'] = (df['Close']-df['Close'].shift(1)).fillna(0)
df['RSI_14D'] = df['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)
df.tail(5)

df['Volume_plain'] = df['Volume'].fillna(0)
df.tail()

#%% Bollinger Bands

def bbands(price, length=30, numsd=2):
    """ returns average, upper band, and lower band"""
    #ave = pd.stats.moments.rolling_mean(price,length)
    ave = price.rolling(window = length, center = False).mean()
    #sd = pd.stats.moments.rolling_std(price,length)
    sd = price.rolling(window = length, center = False).std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return np.round(ave,3), np.round(upband,3), np.round(dnband,3)

df['BB_Middle_Band'], df['BB_Upper_Band'], df['BB_Lower_Band'] = bbands(df['Close'], length=20, numsd=1)
df['BB_Middle_Band'] = df['BB_Middle_Band'].fillna(0)
df['BB_Upper_Band'] = df['BB_Upper_Band'].fillna(0)
df['BB_Lower_Band'] = df['BB_Lower_Band'].fillna(0)
df.tail()

#%% Aroon Oscillator

def aroon(df, tf=25):
    aroonup = []
    aroondown = []
    x = tf
    while x< len(df['Date']):
        aroon_up = ((df['High'][x-tf:x].tolist().index(max(df['High'][x-tf:x])))/float(tf))*100
        aroon_down = ((df['Low'][x-tf:x].tolist().index(min(df['Low'][x-tf:x])))/float(tf))*100
        aroonup.append(aroon_up)
        aroondown.append(aroon_down)
        x+=1
    return aroonup, aroondown


listofzeros = [0] * 25
up, down = aroon(df)
aroon_list = [x - y for x, y in zip(up,down)]
if len(aroon_list)==0:
    aroon_list = [0] * df.shape[0]
    df['Aroon_Oscillator'] = aroon_list
else:
    df['Aroon_Oscillator'] = listofzeros+aroon_list
    
#%% Price Volume Trend

#PVT = [((CurrentClose - PreviousClose) / PreviousClose) x Volume] + PreviousPVT
    
df["PVT"] = (df['Momentum_1D']/ df['Close'].shift(1))*df['Volume']
df["PVT"] = df["PVT"]-df["PVT"].shift(1)
df["PVT"] = df["PVT"].fillna(0)
df.tail()

#%% Acceleration Bands

def abands(df):
    #df['AB_Middle_Band'] = pd.rolling_mean(df['Close'], 20)
    df['AB_Middle_Band'] = df['Close'].rolling(window = 20, center=False).mean()
    # High * ( 1 + 4 * (High - Low) / (High + Low))
    df['aupband'] = df['High'] * (1 + 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Upper_Band'] = df['aupband'].rolling(window=20, center=False).mean()
    # Low *(1 - 4 * (High - Low)/ (High + Low))
    df['adownband'] = df['Low'] * (1 - 4 * (df['High']-df['Low'])/(df['High']+df['Low']))
    df['AB_Lower_Band'] = df['adownband'].rolling(window=20, center=False).mean()
    
abands(df)
df = df.fillna(0)
df.tail()

#%% Dropping Unwanted Columns

columns2Drop = ['Momentum_1D', 'aupband', 'adownband']
df = df.drop(labels = columns2Drop, axis=1)
df.head()
#%% Plotting the results

def plotting(key, predictions, expected, rmse):
    plt.figure()
    plt.title(key)

    y_values = list(expected) + list(predictions)
    y_range = max(y_values) - min(y_values)
    plt.text(6, min(y_values) + 0.2 * y_range, 'RMSE = ' + str(rmse))
    plt.plot(predictions)
    plt.plot(expected)
    plt.legend(['predicted', 'expected'])
    plt.show()


#%%
a,b = df.shape

train_size = int(a*0.8)
test_size = a-train_size

y = df.pop('Close')
date = df.pop('Date')
vol_data = df.pop('Volume')
X = df
trainX = X.head(train_size)
testX = X.tail(test_size)
trainY = y.head(train_size) 
testY = y.tail(test_size)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

#trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
################################# XGBoost Regressor
eval_set = [(testX,testY)]

depth = [3,5,6,7,8,9,10,11]
alpha1 = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10)]#,pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
lambda1 = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10)]#,pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

parameters = []

for num in depth:
    for num1 in alpha1:
        for num2 in lambda1:
            result = []            
            xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,
                             gamma=0,                 
                             learning_rate=0.05,
                             max_depth=num,
                             min_child_weight=1.5,
                             n_estimators=10000,                                                                    
                             reg_alpha=num1,
                             reg_lambda=num2,
                             subsample=0.6,
                             seed=42)
            
            
            xgb_model.fit(trainX, trainY,eval_set=eval_set,verbose=True,early_stopping_rounds=50)
            
            z = xgb_model.predict(testX)
            
            print("MSE for the test part is : ",np.mean((z-testY)**2))
            result.append(num)
            result.append(num1)
            result.append(num2)
            result.append(np.mean((z-testY)**2))
            parameters.append(result)
            print(num,num1,num2)
zpred = xgb_model.predict(X)

print("XGBoost Regression: ",rmse(testY,z) )
plotting("XGBoost Regression", z, testY.values, np.mean((z-testY)**2))

#%%
eval_set = [(testX,testY)]
xgb_model = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.05,
                 max_depth=9,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=16,
                 reg_lambda=64,
                 subsample=0.6,
                 seed=42)


xgb_model.fit(trainX, trainY,eval_set=eval_set,verbose=True,early_stopping_rounds=50)

z = xgb_model.predict(testX)

print("MSE for the test part is : ",np.mean((z-testY)**2))
result.append(num)
result.append(num1)
result.append(num2)
result.append(np.mean((z-testY)**2))
parameters.append(result)
print(num,num1,num2)


print("XGBoost Regression: ",rmse(testY,z) )
plotting("XGBoost Regression", z, testY.values, np.mean((z-testY)**2))


#%% All the Regressors

#################################### Feature Selection
print("Modelling Starts...")
fsr = RandomForestRegressor()

#print trainX
fsr.fit(trainX,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX = modi.transform(trainX)
testX = modi.transform(testX)

#%%
print("=================== Feature Selection Completes")
#################################### Linear Regression
print("================== Linear Regression...")
clf0 = LinearRegression()
param = {"fit_intercept": [True, False],
     "normalize": [False],
     "copy_X": [True, False]}
grid = GridSearchCV(clf0,param,n_jobs=1)
grid.fit(trainX,trainY)
clf0 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR Ends...")
#################################### SVR-Sigmoid
#%%
print("SVR Sig Starts...")
mod = SVR()
g = g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
C = [1]
param = {"kernel": ["sigmoid"],
     "gamma": g,
     "C":C}
random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
random_search.fit(trainX,trainY)
clf2 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig Ends...")
################################### SVR-RBF
#%%
print("SVR Ridge Starts...")
mod = SVR()

g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#g = [pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#print "g", g

C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10)]#,pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

#g = list(np.linspace(0.0000001,8,10000))
param= {'gamma': g,
    'kernel': ['rbf'],
    'C': C}
grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX,trainY)           
clf3 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge Ends ...")
################################# Decision Tree Regressor
#%%
print("DTR Starts...")
dtr = DecisionTreeRegressor()
param_tree = {"max_depth": [3,6,10,12,15,20,None],
	  "min_samples_leaf": sp_randint(1, 30),
	  "criterion": ["mse"],
	  "splitter": ["best","random"],
	  "max_features": ["auto","sqrt",None]}
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX,trainY)
clf4 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR Ends...")
#%%
################################# Random Forest
print("RFR Starts...")
fsr = RandomForestRegressor()
param = {"n_estimators": [100,500,1000,5000],
     "criterion": ["mse"],
     "min_samples_split": [2,10,12,15]}
grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX,trainY)
clf6 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR Ends...")
################################# Adaboost
#%%
print("Adaboost Starts...")
reg = AdaBoostRegressor(clf4)
param = {#"base_estimator":dtr,
     "n_estimators":[100,500],
     "learning_rate":list(np.linspace(0.00000000001,1,100)),
     "loss":["linear", "square", "exponential"]}
gridAB = RandomizedSearchCV(reg,param,n_jobs=1,n_iter=100)
gridAB.fit(trainX,trainY)
clf5 = AdaBoostRegressor(base_estimator=clf4,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost Ends...")
#%%
######################### Ridge Regressor
print("Ridge Starts...")
rdg = Ridge()
para_ridge = {"alpha": list(np.linspace(0.000000001,1000,100)),
	  "fit_intercept": [True, False],
	  "normalize": [True, False],
	  "solver": ["auto"]}			
random_rdg = RandomizedSearchCV(rdg, para_ridge, n_jobs=1, n_iter = 100)
random_rdg.fit(trainX,trainY)
clf7 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])
#%%
print("========================== Ridge Ends...")
#1 Linear Regression
clf0.fit(trainX,trainY)
z0 = clf0.predict(testX)


#2 SVR Sigmoid
clf2.fit(trainX,trainY)
z2 = clf2.predict(testX)

#3 SVR RBF
clf3.fit(trainX,trainY)
z3 = clf3.predict(testX)

#4 Decision Tree Regressor
clf4.fit(trainX,trainY)
z4 = clf4.predict(testX)

#6 AdaBoost Regressor
clf5.fit(trainX,trainY)
z5 = clf5.predict(testX)

#6 RF Regressor
clf6.fit(trainX,trainY)
z6 = clf6.predict(testX)


#6 Ridge Regressor
clf7.fit(trainX,trainY)
z7 = clf7.predict(testX)

err = {clf0:np.mean((z0-testY)**2),
   clf2:np.mean((z2-testY)**2),
   clf3:np.mean((z3-testY)**2),
   clf4:np.mean((z4-testY)**2),
   clf5:np.mean((z5-testY)**2),
   clf6:np.mean((z6-testY)**2),
   clf7:np.mean((z7-testY)**2)}



print("END")
#%% Predictions
for num in range(0,zpred.shape[0]):
    if zpred[num] < 0:
        zpred[num] = 0


#%% Printing the Error Values

print(list(err.values()))
error_vals = list(err.values())
print("Printing the Errors And Plotting for all the values....")
print("Linear Regression: ", error_vals[0] )
plotting("Linear Regression", z0, testY.values, error_vals[0])
print("Support Vector Regression Sigmoid: ", error_vals[1] )
plotting("Support Vector Regression Sigmoid", z2, testY.values, error_vals[1])
print("Support Vector Regression Ridge: ", error_vals[2] )
plotting("Support Vector Regression Ridge", z3, testY.values, error_vals[2])
print("Decision Tree Regression: ", error_vals[3] )
plotting("Decision Tree Regression", z4, testY.values, error_vals[3])
print("AdaBoost Regression: ", error_vals[4] )
plotting("AdaBoost Regression", z5, testY.values, error_vals[4])
print("Random Forest Regression: ", error_vals[5] )
plotting("Random Forest Regression", z6, testY.values, error_vals[5])
print("Ridge Regression: ", error_vals[6] )
plotting("Ridge Regression", z7, testY.values, error_vals[6])
print("XGBoost Regression: ",rmse(testY,z) )
plotting("XGBoost Regression", z, testY.values, np.mean((z-testY)**2))