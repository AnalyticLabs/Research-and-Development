# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:33:41 2019

@author: 91953
"""

#%% importing libraries
"""
ALl the necessary packages are called
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
import xgboost as xgb
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

#%% Functions Defined

"""
To see how the actual is varying with the predicted across the time
Although the Date column is removed for the easy flow and run of the code
"""
def plotting(key, predictions, expected, mse):
    plt.figure()
    plt.title(key)

    y_values = list(expected) + list(predictions)
    y_range = max(y_values) - min(y_values)
    plt.text(6, min(y_values) + 0.2 * y_range, 'MSE = ' + str(mse))
    plt.plot(predictions)
    plt.plot(expected)
    plt.legend(['predicted', 'expected'])
    plt.show()


#%% Inputing the Datasets

datapath = "C:\\Users\\91953\\Desktop\\Analytic Labs\\Teacheron\\Debodeep\\Stock Market\\Data\\DF1\\"

df1 = pd.read_csv(datapath + "AAPL1.csv")
df2 = pd.read_csv(datapath + "AAPL2.csv")
df3 = pd.read_csv(datapath + "AAPL3.csv")
df4 = pd.read_csv(datapath + "AAPL4.csv")

#%% Printing the Columns of Different Data Frames

print("Dataframe 1 Columns......")
print(list(df1.columns.values))
print("Dataframe 1 Columns......")
print(list(df2.columns.values))
print("Dataframe 1 Columns......")
print(list(df3.columns.values))
print("Dataframe 1 Columns......")
print(list(df4.columns.values))
print("End........")

#%% Removing all the unnecessary columns

df1.pop("Adj Close")
df1.pop("Volume")
df1.pop("Label")

print(list(df1.columns.values))

df2.pop('Open')
df2.pop('High')
df2.pop('Low')
df2.pop("Close")
df2.pop('Adj Close')
df2.pop('Volume')
df2.pop('Label')

print(list(df2.columns.values))

df3.pop('Open')
df3.pop('High')
df3.pop('Low')
df3.pop("Close")
df3.pop('Adj Close')
df3.pop('Volume')
df3.pop('Label')

print(list(df3.columns.values))

df4.pop('Open')
df4.pop('High')
df4.pop('Low')
df4.pop("Close")
df4.pop('Adj Close')
df4.pop('Volume')
df4.pop('Label')

print(list(df4.columns.values))

#%% Performing the Merging of the Datasets to form Universal Dataset

df_merged = pd.concat([df1,df2,df3, df4], axis=1, sort=False)

print(list(df_merged.columns.values))

#%% Bringing in the Test and Training part of the ALgorithms

a,b = df1.shape

train_size = int(a*0.8)
test_size = a-train_size

y = df1.pop('Close')

X1 = df1
X2 = df2
X3 = df3
X4 = df4
X_total = df_merged

scaler = StandardScaler()

trainX1 = X1.head(train_size)
testX1 = X1.tail(test_size)
print(trainX1.shape, testX1.shape)

trainX1 = scaler.fit_transform(trainX1)
testX1 = scaler.transform(testX1)

trainX2 = X2.head(train_size)
testX2 = X2.tail(test_size)
print(trainX2.shape, testX2.shape)

trainX2 = scaler.fit_transform(trainX2)
testX2 = scaler.transform(testX2)

trainX3 = X3.head(train_size)
testX3 = X3.tail(test_size)
print(trainX3.shape, testX3.shape)

trainX3 = scaler.fit_transform(trainX3)
testX3 = scaler.transform(testX3)

trainX4 = X4.head(train_size)
testX4 = X4.tail(test_size)
print(trainX4.shape, testX4.shape)

trainX4 = scaler.fit_transform(trainX4)
testX4 = scaler.transform(testX4)

trainX5 = X_total.head(train_size)
testX5 = X_total.tail(test_size)
print(trainX5.shape, testX5.shape)

trainX5 = scaler.fit_transform(trainX5)
testX5 = scaler.transform(testX5)

trainY = y.head(train_size) 
testY = y.tail(test_size)

#%% All the Regressors

#################################### Feature Selection
print("Modelling Starts...")
fsr = RandomForestRegressor()

#print trainX
fsr.fit(trainX1,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX1 = modi.transform(trainX1)
testX1 = modi.transform(testX1)

fsr.fit(trainX2,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX2 = modi.transform(trainX2)
testX2 = modi.transform(testX2)

fsr.fit(trainX3,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX3 = modi.transform(trainX3)
testX3 = modi.transform(testX3)

fsr.fit(trainX4,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX4 = modi.transform(trainX4)
testX4 = modi.transform(testX4)

fsr.fit(trainX5,trainY)
fsr.feature_importances_
modi = SelectFromModel(fsr, prefit=True)
trainX5 = modi.transform(trainX5)
testX5 = modi.transform(testX5)
print("=================== Feature Selection Completes")

#%% Using the XGboost 

eval_set1 = [(testX1,testY)]
eval_set2 = [(testX2,testY)]
eval_set3 = [(testX3,testY)]
eval_set4 = [(testX4,testY)]
eval_set5 = [(testX5,testY)]

xgb_model = xgb.XGBRegressor(colsample_bytree=0.7,
                 gamma=0.841,                 
                 learning_rate=0.05,
                 max_depth=6,
                 min_child_weight=1.5,
                 n_estimators=50000,                                                                    
                 reg_alpha=pow(2,5),
                 #reg_lambda=pow(2,-2),
                 subsample=0.6,
                 seed=42)


xgb_model.fit(trainX1, trainY,eval_set=eval_set1,verbose=True,early_stopping_rounds=50)
z1 = xgb_model.predict(testX1)

xgb_model.fit(trainX2, trainY,eval_set=eval_set2,verbose=True,early_stopping_rounds=50)
z2 = xgb_model.predict(testX1)


xgb_model.fit(trainX3, trainY,eval_set=eval_set3,verbose=True,early_stopping_rounds=50)
z3 = xgb_model.predict(testX3)

xgb_model.fit(trainX4, trainY,eval_set=eval_set4,verbose=True,early_stopping_rounds=50)
z4 = xgb_model.predict(testX4)

xgb_model.fit(trainX5, trainY,eval_set=eval_set5,verbose=True,early_stopping_rounds=50)
z5 = xgb_model.predict(testX5)

print("MSE for the test part1 is : ",np.mean((z1-testY)**2))
print("MSE for the test part2 is : ",np.mean((z2-testY)**2))
print("MSE for the test part3 is : ",np.mean((z3-testY)**2))
print("MSE for the test part4 is : ",np.mean((z4-testY)**2))
print("MSE for the test part5 is : ",np.mean((z5-testY)**2))


#%% Using the Linear Regression

#################################### Linear Regression
print("================== Linear Regression...")
clf0 = LinearRegression()
param = {"fit_intercept": [True, False],
     "normalize": [False],
     "copy_X": [True, False]}
grid = GridSearchCV(clf0,param,n_jobs=1)

grid.fit(trainX1,trainY)
clf01 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR1 Ends...")

grid.fit(trainX2,trainY)
clf02 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR2 Ends...")

grid.fit(trainX3,trainY)
clf03 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR3 Ends...")

grid.fit(trainX4,trainY)
clf04 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR4 Ends...")

grid.fit(trainX5,trainY)
clf05 = LinearRegression(fit_intercept=grid.best_params_["fit_intercept"],normalize=grid.best_params_["normalize"],copy_X=grid.best_params_["copy_X"],n_jobs=-1)
print("================== LR5 Ends...")

#%% Using the SVR Sigmoid

print("SVR Sig Starts...")
mod = SVR()

g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),
     pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),
     pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
C = [1]
param = {"kernel": ["sigmoid"],
     "gamma": g,
     "C":C}
random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=25)
random_search.fit(trainX1,trainY)
clf21 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig1 Ends...")

random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=25)
random_search.fit(trainX2,trainY)
clf22 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig2 Ends...")


random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=25)
random_search.fit(trainX3,trainY)
clf23 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig3 Ends...")


random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=25)
random_search.fit(trainX4,trainY)
clf24 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig4 Ends...")


random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=25)
random_search.fit(trainX5,trainY)
clf25 = SVR(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print("==================== SVR Sig5 Ends...")

#%%

print("SVR Ridge Starts...")
mod = SVR()

g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),
     pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#g = [pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#print "g", g

C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),
     pow(2,7),pow(2,8),pow(2,9),pow(2,10)]#,pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

#g = list(np.linspace(0.0000001,8,10000))
param= {'gamma': g,
    'kernel': ['rbf'],
    'C': C}
grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX1,trainY)           
clf31 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge1 Ends ...")

grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX2,trainY)           
clf32 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge2 Ends ...")

grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX3,trainY)           
clf33 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge3 Ends ...")

grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX4,trainY)           
clf34 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge4 Ends ...")

grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=50)
grid_search.fit(trainX5,trainY)           
clf35 = SVR(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 
print("===================== SVR Ridge5 Ends ...")

#%%

print("DTR Starts...")
dtr = DecisionTreeRegressor()
param_tree = {"max_depth": [3,6,10,12,15,20,None],
	  "min_samples_leaf": sp_randint(1, 30),
	  "criterion": ["mse"],
	  "splitter": ["best","random"],
	  "max_features": ["auto","sqrt",None]}
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX1,trainY)
clf41 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR1 Ends...")
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX2,trainY)
clf42 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR2 Ends...")
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX3,trainY)
clf43 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR3 Ends...")
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX4,trainY)
clf44 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR4 Ends...")
	  
gridDT = RandomizedSearchCV(dtr,param_tree,n_jobs=1,n_iter=100)
gridDT.fit(trainX5,trainY)
clf45 = DecisionTreeRegressor(criterion=gridDT.best_params_["criterion"],splitter=gridDT.best_params_["splitter"],max_features=gridDT.best_params_["max_features"],max_depth=gridDT.best_params_["max_depth"],min_samples_leaf=gridDT.best_params_["min_samples_leaf"])
print("======================= DTR5 Ends...")

#%%################################# Random Forest
print("RFR Starts...")
fsr = RandomForestRegressor()
param = {"n_estimators": [100,500,1000,5000],
     "criterion": ["mse"],
     "min_samples_split": [2,10,12,15]}
grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX1,trainY)
clf61 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR1 Ends...")

grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX2,trainY)
clf62 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR2 Ends...")

grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX3,trainY)
clf63 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR3 Ends...")

grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX4,trainY)
clf64 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR4 Ends...")

grid = GridSearchCV(fsr,param,n_jobs=1)
grid.fit(trainX5,trainY)
clf65 = RandomForestRegressor(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],min_samples_split=grid.best_params_["min_samples_split"])
print("========================= RFR5 Ends...")

#%%

print("Adaboost Starts...")
reg = AdaBoostRegressor(clf41)
param = {#"base_estimator":dtr,
     "n_estimators":[100,500],
     "learning_rate":list(np.linspace(0.00000000001,1,100)),
     "loss":["linear", "square", "exponential"]}
gridAB = RandomizedSearchCV(reg,param,n_jobs=1,n_iter=100)
gridAB.fit(trainX1,trainY)
clf51 = AdaBoostRegressor(base_estimator=clf41,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost1 Ends...")

gridAB.fit(trainX2,trainY)
clf52 = AdaBoostRegressor(base_estimator=clf41,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost2 Ends...")

gridAB.fit(trainX3,trainY)
clf53 = AdaBoostRegressor(base_estimator=clf41,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost3 Ends...")

gridAB.fit(trainX4,trainY)
clf54 = AdaBoostRegressor(base_estimator=clf41,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost4 Ends...")

gridAB.fit(trainX5,trainY)
clf55 = AdaBoostRegressor(base_estimator=clf41,n_estimators=gridAB.best_params_["n_estimators"],learning_rate=gridAB.best_params_["learning_rate"],loss=gridAB.best_params_["loss"])
print("========================= Adaboost5 Ends...")

#%%######################### Ridge Regressor
print("Ridge Starts...")
rdg = Ridge()
para_ridge = {"alpha": list(np.linspace(0.000000001,1000,100)),
	  "fit_intercept": [True, False],
	  "normalize": [True, False],
	  "solver": ["auto"]}			
random_rdg = RandomizedSearchCV(rdg, para_ridge, n_jobs=1, n_iter = 100)
random_rdg.fit(trainX1,trainY)
clf71 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

random_rdg.fit(trainX2,trainY)
clf72 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

random_rdg.fit(trainX3,trainY)
clf73 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

random_rdg.fit(trainX4,trainY)
clf74 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])

random_rdg.fit(trainX5,trainY)
clf75 = Ridge(alpha=random_rdg.best_params_["alpha"],fit_intercept=random_rdg.best_params_["fit_intercept"],normalize=random_rdg.best_params_["normalize"],solver=random_rdg.best_params_["solver"])


#%% MSE Evaluation

#1 Linear Regression
clf01.fit(trainX1,trainY)
z01 = clf01.predict(testX1)

clf02.fit(trainX2,trainY)
z02 = clf02.predict(testX2)

clf03.fit(trainX3,trainY)
z03 = clf03.predict(testX3)

clf04.fit(trainX4,trainY)
z04 = clf04.predict(testX4)

clf05.fit(trainX5,trainY)
z05 = clf05.predict(testX5)

#%%
#2 SVR Sigmoid
clf21.fit(trainX1,trainY)
z21 = clf21.predict(testX1)

clf22.fit(trainX2,trainY)
z22 = clf22.predict(testX2)

clf23.fit(trainX3,trainY)
z23 = clf23.predict(testX3)

clf24.fit(trainX4,trainY)
z24 = clf24.predict(testX4)

clf25.fit(trainX5,trainY)
z25 = clf25.predict(testX5)

#%%
#3 SVR Ridge
clf31.fit(trainX1,trainY)
z31 = clf31.predict(testX1)

clf32.fit(trainX2,trainY)
z32 = clf32.predict(testX2)

clf33.fit(trainX3,trainY)
z33 = clf33.predict(testX3)

clf34.fit(trainX4,trainY)
z34 = clf34.predict(testX4)

clf35.fit(trainX5,trainY)
z35 = clf35.predict(testX5)

#%%
#4 Decision Trees Regression

clf41.fit(trainX1,trainY)
z41 = clf41.predict(testX1)

clf42.fit(trainX2,trainY)
z42 = clf42.predict(testX2)

clf43.fit(trainX3,trainY)
z43 = clf43.predict(testX3)

clf44.fit(trainX4,trainY)
z44 = clf44.predict(testX4)

clf45.fit(trainX5,trainY)
z45 = clf45.predict(testX5)

#%%
# Random Forest Regression

clf61.fit(trainX1,trainY)
z61 = clf61.predict(testX1)

clf62.fit(trainX2,trainY)
z62 = clf62.predict(testX2)

clf63.fit(trainX3,trainY)
z63 = clf63.predict(testX3)

clf64.fit(trainX4,trainY)
z64 = clf64.predict(testX4)

clf65.fit(trainX5,trainY)
z65 = clf65.predict(testX5)

#%%
# AdaBoost Regression

clf51.fit(trainX1,trainY)
z51 = clf51.predict(testX1)

clf52.fit(trainX2,trainY)
z52 = clf52.predict(testX2)

clf53.fit(trainX3,trainY)
z53 = clf53.predict(testX3)

clf54.fit(trainX4,trainY)
z54 = clf54.predict(testX4)

clf55.fit(trainX5,trainY)
z55 = clf55.predict(testX5)

#%% 
# Ridge Regression

clf71.fit(trainX1,trainY)
z71 = clf71.predict(testX1)

clf72.fit(trainX2,trainY)
z72 = clf72.predict(testX2)

clf73.fit(trainX3,trainY)
z73 = clf73.predict(testX3)

clf74.fit(trainX4,trainY)
z74 = clf74.predict(testX4)

clf75.fit(trainX5,trainY)
z75 = clf75.predict(testX5)


#%% 

err = []
dataset_used = []
forecast_values = []
model = []

#%%

err_LR = {"D01":np.mean((z01-testY)**2),
          "D02":np.mean((z02-testY)**2),
          "D03":np.mean((z03-testY)**2),
          "D04":np.mean((z04-testY)**2),
          "D05":np.mean((z05-testY)**2)}

err.append((z01-testY)**2)
err.append((z02-testY)**2)
err.append((z03-testY)**2)
err.append((z04-testY)**2)
err.append((z05-testY)**2)

dataset_used.append("DO1")
dataset_used.append("DO2")
dataset_used.append("DO3")
dataset_used.append("DO4")
dataset_used.append("DO5")

forecast_values.append(z01)
forecast_values.append(z02)
forecast_values.append(z03)
forecast_values.append(z04)
forecast_values.append(z05)

model.append("LR")
model.append("LR")
model.append("LR")
model.append("LR")
model.append("LR")

print(list(err_LR.values()))
error_vals = list(err_LR.values())
print("Printing the Errors And Plotting for all the values....")

print("Linear Regression D01: ", error_vals[0] )
plotting("Linear Regression for D01", z01, testY.values, error_vals[0])

print("Linear Regression D02: ", error_vals[1] )
plotting("Linear Regression for D02", z02, testY.values, error_vals[1])

print("Linear Regression D03: ", error_vals[2] )
plotting("Linear Regression for D03", z03, testY.values, error_vals[2])

print("Linear Regression D04: ", error_vals[3] )
plotting("Linear Regression for D04", z04, testY.values, error_vals[3])

print("Linear Regression D05: ", error_vals[4] )
plotting("Linear Regression for D05", z05, testY.values, error_vals[4])

#%% 

err_SVR_Sig = {"D21":np.mean((z21-testY)**2),
          "D22":np.mean((z22-testY)**2),
          "D23":np.mean((z23-testY)**2),
          "D24":np.mean((z24-testY)**2),
          "D25":np.mean((z25-testY)**2)}

err.append((z21-testY)**2)
err.append((z22-testY)**2)
err.append((z23-testY)**2)
err.append((z24-testY)**2)
err.append((z25-testY)**2)

dataset_used.append("D21")
dataset_used.append("D22")
dataset_used.append("D23")
dataset_used.append("D24")
dataset_used.append("D25")

forecast_values.append(z21)
forecast_values.append(z22)
forecast_values.append(z23)
forecast_values.append(z24)
forecast_values.append(z25)

model.append("SVR_Sig")
model.append("SVR_Sig")
model.append("SVR_Sig")
model.append("SVR_Sig")
model.append("SVR_Sig")


print(list(err_SVR_Sig.values()))
error_vals = list(err_SVR_Sig.values())
print("Printing the Errors And Plotting for all the values....")

print("SVR-Sigmoid D21: ", error_vals[0] )
plotting("SVR-Sigmoid for D21", z21, testY.values, error_vals[0])

print("SVR-Sigmoid D22: ", error_vals[1] )
plotting("SVR-Sigmoid for D22", z22, testY.values, error_vals[1])

print("SVR-Sigmoid D23: ", error_vals[2] )
plotting("SVR-Sigmoid for D23", z23, testY.values, error_vals[2])

print("SVR-Sigmoid D24: ", error_vals[3] )
plotting("SVR-Sigmoid for D24", z24, testY.values, error_vals[3])

print("SVR-Sigmoid D25: ", error_vals[4] )
plotting("SVR-Sigmoid for D25", z25, testY.values, error_vals[4])

#%% 

err_SVR_Rig = {"D31":np.mean((z31-testY)**2),
          "D32":np.mean((z32-testY)**2),
          "D33":np.mean((z33-testY)**2),
          "D34":np.mean((z34-testY)**2),
          "D35":np.mean((z35-testY)**2)}

err.append((z31-testY)**2)
err.append((z32-testY)**2)
err.append((z33-testY)**2)
err.append((z34-testY)**2)
err.append((z35-testY)**2)

dataset_used.append("D31")
dataset_used.append("D32")
dataset_used.append("D33")
dataset_used.append("D34")
dataset_used.append("D35")

forecast_values.append(z31)
forecast_values.append(z32)
forecast_values.append(z33)
forecast_values.append(z34)
forecast_values.append(z35)

model.append("SVR_RBF")
model.append("SVR_RBF")
model.append("SVR_RBF")
model.append("SVR_RBF")
model.append("SVR_RBF")

print(list(err_SVR_Rig.values()))
error_vals = list(err_SVR_Rig.values())
print("Printing the Errors And Plotting for all the values....")

print("SVR-Ridge D31: ", error_vals[0] )
plotting("SVR-Ridge for D31", z31, testY.values, error_vals[0])

print("SVR-Ridge D32: ", error_vals[1] )
plotting("SVR-Ridge for D32", z32, testY.values, error_vals[1])

print("SVR-Ridge D33: ", error_vals[2] )
plotting("SVR-Ridge for D33", z33, testY.values, error_vals[2])

print("SVR-Ridge D34: ", error_vals[3] )
plotting("SVR-Ridge for D34", z34, testY.values, error_vals[3])

print("SVR-Ridge D35: ", error_vals[4] )
plotting("SVR-Ridge for D35", z35, testY.values, error_vals[4])

#%% Decision Trees Regrssion

err_DTR = {"D41":np.mean((z41-testY)**2),
          "D42":np.mean((z42-testY)**2),
          "D43":np.mean((z43-testY)**2),
          "D44":np.mean((z44-testY)**2),
          "D45":np.mean((z45-testY)**2)}

err.append((z41-testY)**2)
err.append((z42-testY)**2)
err.append((z43-testY)**2)
err.append((z44-testY)**2)
err.append((z45-testY)**2)

dataset_used.append("D41")
dataset_used.append("D42")
dataset_used.append("D43")
dataset_used.append("D44")
dataset_used.append("D45")

forecast_values.append(z41)
forecast_values.append(z42)
forecast_values.append(z43)
forecast_values.append(z44)
forecast_values.append(z45)

model.append("DTR")
model.append("DTR")
model.append("DTR")
model.append("DTR")
model.append("DTR")

print(list(err_DTR.values()))
error_vals = list(err_DTR.values())
print("Printing the Errors And Plotting for all the values....")

print("DTR D41: ", error_vals[0] )
plotting("DTR for D41", z41, testY.values, error_vals[0])

print("DTR D42: ", error_vals[1] )
plotting("DTR for D42", z42, testY.values, error_vals[1])

print("DTR D43: ", error_vals[2] )
plotting("DTR for D43", z43, testY.values, error_vals[2])

print("DTR D44: ", error_vals[3] )
plotting("DTR for D44", z44, testY.values, error_vals[3])

print("DTR D45: ", error_vals[4] )
plotting("DTR for D45", z45, testY.values, error_vals[4])

#%% Random Forest Regression

err_RFR = {"D61":np.mean((z61-testY)**2),
          "D62":np.mean((z62-testY)**2),
          "D63":np.mean((z63-testY)**2),
          "D64":np.mean((z64-testY)**2),
          "D65":np.mean((z65-testY)**2)}

err.append((z61-testY)**2)
err.append((z62-testY)**2)
err.append((z63-testY)**2)
err.append((z64-testY)**2)
err.append((z65-testY)**2)

dataset_used.append("D61")
dataset_used.append("D62")
dataset_used.append("D63")
dataset_used.append("D64")
dataset_used.append("D65")

forecast_values.append(z61)
forecast_values.append(z62)
forecast_values.append(z63)
forecast_values.append(z64)
forecast_values.append(z65)

model.append("RFR")
model.append("RFR")
model.append("RFR")
model.append("RFR")
model.append("RFR")


print(list(err_RFR.values()))
error_vals = list(err_RFR.values())
print("Printing the Errors And Plotting for all the values....")

print("RFR D61: ", error_vals[0] )
plotting("RFR for D61", z61, testY.values, error_vals[0])

print("RFR D62: ", error_vals[1] )
plotting("RFR for D62", z62, testY.values, error_vals[1])

print("RFR D63: ", error_vals[2] )
plotting("RFR for D63", z63, testY.values, error_vals[2])

print("RFR D64: ", error_vals[3] )
plotting("RFR for D64", z64, testY.values, error_vals[3])

print("RFR D65: ", error_vals[4] )
plotting("RFR for D65", z65, testY.values, error_vals[4])

#%% Adaboost Regression

err_ADR = {"D51":np.mean((z51-testY)**2),
          "D52":np.mean((z52-testY)**2),
          "D53":np.mean((z53-testY)**2),
          "D54":np.mean((z54-testY)**2),
          "D55":np.mean((z55-testY)**2)}

err.append((z51-testY)**2)
err.append((z52-testY)**2)
err.append((z53-testY)**2)
err.append((z54-testY)**2)
err.append((z55-testY)**2)

dataset_used.append("D51")
dataset_used.append("D52")
dataset_used.append("D53")
dataset_used.append("D54")
dataset_used.append("D55")

forecast_values.append(z51)
forecast_values.append(z52)
forecast_values.append(z53)
forecast_values.append(z54)
forecast_values.append(z55)

model.append("ADR")
model.append("ADR")
model.append("ADR")
model.append("ADR")
model.append("ADR")


print(list(err_ADR.values()))
error_vals = list(err_ADR.values())
print("Printing the Errors And Plotting for all the values....")

print("ADR D51: ", error_vals[0] )
plotting("ADR for D51", z51, testY.values, error_vals[0])

print("ADR D52: ", error_vals[1] )
plotting("ADR for D52", z52, testY.values, error_vals[1])

print("ADR D53: ", error_vals[2] )
plotting("ADR for D53", z53, testY.values, error_vals[2])

print("ADR D54: ", error_vals[3] )
plotting("ADR for D54", z54, testY.values, error_vals[3])

print("ADR D55: ", error_vals[4] )
plotting("ADR for D55", z55, testY.values, error_vals[4])

#%% Ridge Regression

err_RR = {"D71":np.mean((z71-testY)**2),
          "D72":np.mean((z72-testY)**2),
          "D73":np.mean((z73-testY)**2),
          "D74":np.mean((z74-testY)**2),
          "D75":np.mean((z75-testY)**2)}

err.append((z71-testY)**2)
err.append((z72-testY)**2)
err.append((z73-testY)**2)
err.append((z74-testY)**2)
err.append((z75-testY)**2)

dataset_used.append("D71")
dataset_used.append("D72")
dataset_used.append("D73")
dataset_used.append("D74")
dataset_used.append("D75")

forecast_values.append(z71)
forecast_values.append(z72)
forecast_values.append(z73)
forecast_values.append(z74)
forecast_values.append(z75)

model.append("Ridge")
model.append("Ridge")
model.append("Ridge")
model.append("Ridge")
model.append("Ridge")


print(list(err_RR.values()))
error_vals = list(err_RR.values())
print("Printing the Errors And Plotting for all the values....")

print("RR D71: ", error_vals[0] )
plotting("RR for D71", z71, testY.values, error_vals[0])

print("RR D72: ", error_vals[1] )
plotting("RR for D72", z72, testY.values, error_vals[1])

print("RR D73: ", error_vals[2] )
plotting("RR for D73", z73, testY.values, error_vals[2])

print("RR D74: ", error_vals[3] )
plotting("RR for D74", z74, testY.values, error_vals[3])

print("RR D75: ", error_vals[4] )
plotting("RR for D75", z75, testY.values, error_vals[4])

#%% XGBoost
print("Printing the Errors And Plotting for all the values....")

print("XGB MSE for the test part1 is : ",np.mean((z1-testY)**2))
plotting("XGB for D1", z1, testY.values,np.mean((z1-testY)**2) )

print("XGB MSE for the test part2 is : ",np.mean((z2-testY)**2))
plotting("XGB for D2", z2, testY.values,np.mean((z2-testY)**2) )


print("XGB MSE for the test part3 is : ",np.mean((z3-testY)**2))
plotting("XGB for D3", z3, testY.values,np.mean((z3-testY)**2) )


print("XGB MSE for the test part4 is : ",np.mean((z4-testY)**2))
plotting("XGB for D4", z4, testY.values,np.mean((z4-testY)**2) )


print("XGB MSE for the test part5 is : ",np.mean((z5-testY)**2))
plotting("XGB for D5", z5, testY.values,np.mean((z5-testY)**2) )

#%%
err.append((z1-testY)**2)
err.append((z2-testY)**2)
err.append((z3-testY)**2)
err.append((z4-testY)**2)
err.append((z5-testY)**2)

dataset_used.append("D1")
dataset_used.append("D2")
dataset_used.append("D3")
dataset_used.append("D4")
dataset_used.append("D5")

forecast_values.append(z1)
forecast_values.append(z2)
forecast_values.append(z3)
forecast_values.append(z4)
forecast_values.append(z5)

model.append("XgBoost")
model.append("XgBoost")
model.append("XgBoost")
model.append("XgBoost")
model.append("XgBoost")


#%% Ensemble Part

best_model_dict = {}

print("Printing the Best Error of Linear Regression...: ")
print(min(err_LR.values()))

best_model = [(value, key) for key, value in err_LR.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["LR"] = min(best_model)[0]

print("Printing the Best Error of SVR Sigmoid...: ")
print(min(err_SVR_Sig.values()))

best_model = [(value, key) for key, value in err_SVR_Sig.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["SVR_Sig"] = min(best_model)[0]

print("Printing the Best Error of SVR Ridge...: ")
print(min(err_SVR_Rig.values()))

best_model = [(value, key) for key, value in err_SVR_Rig.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["SVR_Rig"] = min(best_model)[0]

print("Printing the Best Error of DTR...: ")
print(min(err_DTR.values()))

best_model = [(value, key) for key, value in err_DTR.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["DTR"] = min(best_model)[0]

print("Printing the Best Error of ADR...: ")
print(min(err_ADR.values()))

best_model = [(value, key) for key, value in err_ADR.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["ADR"] = min(best_model)[0]

print("Printing the Best Error of RFR...: ")
print(min(err_RFR.values()))

best_model = [(value, key) for key, value in err_RFR.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["RFR"] = min(best_model)[0]

print("Printing the Best Error of RR...: ")
print(min(err_RR.values()))

best_model = [(value, key) for key, value in err_RR.items()]
print ("BEST MODEL: ",min(best_model)[1])

best_model_dict["RR"] = min(best_model)[0]

#%%
forecast_dataframe = pd.DataFrame({"Model Used": model, "Dataset Used": dataset_used, "MSE": err})
forecast_df = pd.DataFrame(forecast_values)
forecast_dataframe = pd.concat([forecast_dataframe,forecast_df],axis = 1)

#%% Best Model Ensemble

#TODO: Automate it 
 
ML_model_forecast = z35
TS_model_forecast = z05

mse_ml = np.mean((z35-testY)**2)
mse_ts = np.mean((z05-testY)**2)

w_ml = (1/mse_ml)/(1/mse_ts + 1/mse_ml)
w_ts = (1/mse_ts)/(1/mse_ts + 1/mse_ml)

forecast_ensemble = w_ml*ML_model_forecast + w_ts*TS_model_forecast

#%% Deep Dive Ensemble

print("MSE for the EnSemble is : ",np.mean((forecast_ensemble-testY)**2))
plotting("Ensemble Results Plot", forecast_ensemble, testY.values,np.mean((forecast_ensemble-testY)**2) )

print("MSE for the TS is : ",np.mean((TS_model_forecast-testY)**2))
plotting("TS Results Plot", TS_model_forecast, testY.values,np.mean((TS_model_forecast-testY)**2) )

print("MSE for the Machine Learning is : ",np.mean((ML_model_forecast-testY)**2))
plotting("ML Results Plot", ML_model_forecast, testY.values,np.mean((ML_model_forecast-testY)**2) )
