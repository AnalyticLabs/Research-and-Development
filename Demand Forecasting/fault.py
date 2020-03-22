# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 20:42:40 2020

@author: 91953
"""

#%% Importing the Packages
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
#%%
def accuracy(Yp,Ya):
	a = len(Yp)
	count = 0
	for num in range(0,a):
		if Yp[num] == Ya[num]:
			count += 1
	#print a
	return count/float(a)


#%%
data = pd.read_csv("C:\\Users\\91953\\Desktop\\Analytic Labs\\RnD\\Demand Forecasting\\VI_Dataset.csv")
target = data["Category"]

data.pop("Category")
training = data
#%%
Xtrain, Xtest, Ytrain, Ytest = train_test_split(training.values,target.values, test_size=0.3, random_state=42)

#%%
##################################### Feature Selection ###########################################
start_time = time.time()
mod = RandomForestClassifier()
#param = {"n_estimators": [100],
#	 "criterion": ["gini","entropy"],
#	 "max_features": ["auto","sqrt","log2",None],
#	 "oob_score": [True, False]}

#grid = GridSearchCV(mod, param, n_jobs=1)
#grid.fit(Xtrain,Ytrain)

#clf5 = RandomForestClassifier(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],max_features=grid.best_params_["max_features"],oob_score=grid.best_params_["oob_score"])

mod.fit(Xtrain,Ytrain)
mod.feature_importances_
modi = SelectFromModel(mod, prefit=True)
Xtrain = modi.transform(Xtrain)
Xtest = modi.transform(Xtest)
print("--- %s seconds ---" % (time.time() - start_time))
print("Feature Selection Code Completes......")
#X2 = modi.transform(X2)

#####################################  Classification Models ######################################
"""
################################################################################################### SVM
start_time = time.time()
mod = SVC()

#################################### SVR-Sigmoid

g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]

C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

param = {"kernel": ["sigmoid"],
     "gamma": g,
     "C":C}
random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
random_search.fit(Xtrain,Ytrain)
clf0 = SVC(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print ("Check 1")
print("--- %s seconds ---" % (time.time() - start_time))
print("SVR-Sig------")
################################### SVR-RBF
start_time = time.time()
param= {'gamma': g,
    'kernel': ['rbf'],
    'C': C}
grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
grid_search.fit(Xtrain,Ytrain)           
clf1 = SVC(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"]) 


clf0.fit(Xtrain,Ytrain)
z0=clf0.predict(Xtest)
print (z0,Ytest)
clf1.fit(Xtrain,Ytrain)
z1=clf1.predict(Xtest)
print("SVM-Sigmoid: ",accuracy(z0,Ytest))
print("SVM-RBF: ",accuracy(z1,Ytest))
print("--- %s seconds ---" % (time.time() - start_time))
print("SVR-RBF------")
print ("Check 2")
"""
################################################################################################### Logistic Regression
start_time = time.time()
g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]

C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

mod = LogisticRegression()
param = {"penalty":['l1'],
	 "dual": [False],
	 "C":C,
	 "fit_intercept": [True, False],
	 "solver": ["liblinear"]}

grid = GridSearchCV(mod,param,n_jobs=1)
grid.fit(Xtrain,Ytrain)

clf2 = LogisticRegression(penalty=grid.best_params_["penalty"],dual=grid.best_params_["dual"],C=grid.best_params_["C"],fit_intercept=grid.best_params_["fit_intercept"],solver=grid.best_params_["solver"])
print("--- %s seconds ---" % (time.time() - start_time))
print("LR-L1------")

start_time = time.time()
param = {"penalty":['l2'],
	 "dual": [False],
	 "C":C,
	 "fit_intercept": [True, False],
	 "solver": ["newton-cg", "lbfgs", "liblinear", "sag"]}

grid = GridSearchCV(mod,param,n_jobs=1)
grid.fit(Xtrain,Ytrain)

clf3 = LogisticRegression(penalty=grid.best_params_["penalty"],dual=grid.best_params_["dual"],C=grid.best_params_["C"],fit_intercept=grid.best_params_["fit_intercept"],solver=grid.best_params_["solver"])


clf2.fit(Xtrain,Ytrain)
z2 = clf2.predict(Xtest)

clf3.fit(Xtrain,Ytrain)
z3 = clf3.predict(Xtest)
print ("Logistic l1: ",accuracy(z2,Ytest))
print ("Logistic l2: ",accuracy(z3,Ytest))

print("--- %s seconds ---" % (time.time() - start_time))
print("LR-L2------")
print ("check 3")
################################################################################################### Decision Trees
start_time = time.time()
mod = DecisionTreeClassifier()
param = {"criterion": ["gini","entropy"],
	 "splitter": ["best","random"],
	 "max_features": ["auto","sqrt","log2",None],
	 "presort": [True, False]}

grid = GridSearchCV(mod,param,n_jobs=1)
grid.fit(Xtrain,Ytrain)

clf4 = DecisionTreeClassifier(criterion=grid.best_params_["criterion"],splitter=grid.best_params_["splitter"],max_features=grid.best_params_["max_features"],presort=grid.best_params_["presort"])

clf4.fit(Xtrain,Ytrain)
z4 = clf4.predict(Xtest)
print("Decition trees Claasifier: ", accuracy(z4,Ytest))

print("--- %s seconds ---" % (time.time() - start_time))
print("Decision Trees ------")

print ("check 4")
################################################################################################### Random Forest
start_time = time.time()
mod = RandomForestClassifier()
param = {"n_estimators": [100,500,1000,4000],
	 "criterion": ["gini","entropy"],
	 "max_features": ["auto","sqrt","log2",None],
	 "oob_score": [True, False]}

grid = GridSearchCV(mod, param, n_jobs=1)
grid.fit(Xtrain,Ytrain)

clf5 = RandomForestClassifier(n_estimators=grid.best_params_["n_estimators"],criterion=grid.best_params_["criterion"],max_features=grid.best_params_["max_features"],oob_score=grid.best_params_["oob_score"])

clf5.fit(Xtrain,Ytrain)
z5 = clf5.predict(Xtest)
print("Random Forest: ",accuracy(z5,Ytest))

print("--- %s seconds ---" % (time.time() - start_time))
print("Random Forest ------")
print ("check 5")
################################################################################################### Naive Bayes
start_time = time.time()
clf6 = GaussianNB()
clf6.fit(Xtrain,Ytrain)
z6 = clf6.predict(Xtest)


mod = MultinomialNB()
param = {"alpha": np.linspace(0.00001,1,1000),
	 "fit_prior": [True, False]}
grid = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
grid.fit(Xtrain,Ytrain)
clf7 = MultinomialNB(alpha=grid.best_params_["alpha"],fit_prior=grid.best_params_["fit_prior"])
clf7.fit(Xtrain,Ytrain)

z7 = clf7.predict(Xtest)
print ("NB-Gaussian: ",accuracy(z6,Ytest))
print ("Multi NB: ",accuracy(z7,Ytest))

print("--- %s seconds ---" % (time.time() - start_time))
print("GNB ------")
print ("check 6")

################################################################################################### KNN


start_time = time.time()
mod = KNeighborsClassifier()
param = {"n_neighbors": range(1,100,1),
	 "weights": ["uniform", "distance"],
	 "algorithm": ["auto","ball_tree","kd_tree","brute"],
	 "p":[1,2]}
grid = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
grid.fit(Xtrain,Ytrain)
clf11 = KNeighborsClassifier(n_neighbors=grid.best_params_["n_neighbors"],weights=grid.best_params_["weights"],algorithm=grid.best_params_["algorithm"],p=grid.best_params_["p"])
print("--- %s seconds ---" % (time.time() - start_time))
clf11.fit(Xtrain, Ytrain)
z11 = clf11.predict(Xtest)
print("KNN: ",accuracy(z11,Ytest))
print ("check 7")
################################################################################################### Bagging

start_time = time.time()
mod = BaggingClassifier(base_estimator=clf4)
param = {"n_estimators": [100,500,1000,4000]}
grid = GridSearchCV(mod,param,n_jobs=1)
grid.fit(Xtrain,Ytrain)
clf13 = BaggingClassifier(base_estimator=clf4,n_estimators=grid.best_params_["n_estimators"])
clf13.fit(Xtrain,Ytrain)
z13=clf13.predict(Xtest)
print("Bagging: ",accuracy(z13,Ytest))

print("--- %s seconds ---" % (time.time() - start_time))
print("Bassing-DT ------")
print ("check 8")
################################################################################################### Regularization

start_time = time.time()
mod = RidgeClassifier()
params ={"alpha":np.linspace(0.00001,1000,10000),
	 "solver": ["auto","svd","cholesky","lsqr","sparse_cg","sag"],
	 "normalize": [True,False]}

grid = RandomizedSearchCV(mod,params,n_jobs=1,n_iter=100)
grid.fit(Xtrain,Ytrain)
clf14 = RidgeClassifier(alpha=grid.best_params_["alpha"],solver=grid.best_params_["solver"],normalize=grid.best_params_["normalize"])
clf14.fit(Xtrain,Ytrain)
print ("check 9")
z14 = clf14.predict(Xtest)
print("Ridge: ",accuracy(z14,Ytest))
print("--- %s seconds ---" % (time.time() - start_time))
print("Ridge ------")