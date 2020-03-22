# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:31:16 2018

@author: ndasadhi
"""

import pandas as pd
import time
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import Imputer, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from imblearn.over_sampling import SMOTE 
#from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


np.random.seed(7)


def accuracy(Yp,Ya):
	a = len(Yp)
	count = 0
	for num in range(0,a):
		if Yp[num] == Ya[num]:
			count += 1
	#print a
	return count/float(a)

import warnings
warnings.filterwarnings("ignore")
    
#outliers treatment
def outlier_treat(df):
    mean = df.mean()
    std = df.std()
    upper_limit = mean + 3*std
    lower_limit = mean - 3*std
    dataset = df.values
    for num in range(0,df.shape[0]):
        #print(dataset[num])
        if dataset[num] > upper_limit:
            dataset[num] = upper_limit
        elif dataset[num] <= lower_limit:
            dataset[num] = lower_limit
        else:
            dataset[num] = dataset[num]
    return pd.DataFrame(dataset)

#normalization
def minmax_scaling(df):
    scale = MinMaxScaler()
    scale.fit(df)
    df = scale.fit_transform(df)
    return df

#class imbalace treatment
def smote_learn(X,y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_sample(X,y)
    return X_res,y_res

##############################################################################
##############################################################################

#m = raw_input("Enter the file name: ")
df = pd.read_csv("C:\\Users\\ndasadhi\\Desktop\\Competitions\\Loan\\train.csv")
t1 = df.pop("Loan_ID")
df_test = pd.read_csv("C:\\Users\\ndasadhi\\Desktop\\Competitions\\Loan\\test.csv")
t2 = df_test.pop("Loan_ID")
a,b = df_test.shape

Y = df.pop("Loan_Status")

label = LabelEncoder()
Y = label.fit_transform(Y)

df = df.append(df_test, ignore_index=True)

features = list(df.columns.values)
cat_feature = ["Gender","Married","Dependents","Education","Self_Employed","Loan_Amount_Term","Credit_History","Property_Area"]
num_feature = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]
print(features)
print(df.describe())

#a,b = Xtrain.shape
#df['LOSS_ACC_ST_PROV'] = df['LOSS_ACC_ST_PROV'].apply(str)
##################################### Categorical Encoding
for num in cat_feature:
    key = list(df[num].value_counts().keys())
    if sum(df[num].value_counts()) != a:
        max1 = max(df[num].value_counts())
        for num1 in key:
            if df[num].value_counts()[num1] == max1:
                name = num1
                break
        df[num] = df[num].fillna(name)
    label = LabelEncoder()
    df[num] = pd.DataFrame(label.fit_transform(df[num]))
    #print (df[num])

for num in num_feature:
    if df[num].describe()[0] != a:
        mean1 = df[num].mean()
        df[num] = df[num].fillna(mean1)
        df[num] = outlier_treat(df[num])
    else:
        #print(df[num].describe())
        df[num] = outlier_treat(df[num])
        #print(df[num].describe())
        
###################################################### Temporary code to change the 0's missing Values
df.fillna(0)       
print("check1")

#Y = df.pop("Loan_Status")
#t= df.pop("Loan_ID")

a1,b1 = df.shape
###################################################### Below imp
#x_col = ['Gender','Married','Education','Self_Employed','Property_Area']
cat_feature = ["Gender","Married","Dependents","Education","Self_Employed","Loan_Amount_Term","Credit_History","Property_Area"]


hotencoder = OneHotEncoder(sparse=False)
onehot = hotencoder.fit_transform(df[cat_feature])
onehotpd = pd.DataFrame(onehot)
poly = PolynomialFeatures(12)
dfnum = pd.DataFrame(poly.fit_transform(df[num_feature]))
X1 = pd.concat([onehotpd,dfnum],axis=1)

X11 = X1.head(a1-a)

X12 = X1.tail(a)


Xres,Yres = smote_learn(X11,Y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X11, Y, test_size=0.3, random_state=42)
print("Nimai 1")


################################################################################################## XGBoost
start_time = time.time()

start_time = time.time()
mod = SVC()

#################################### SVR-Sigmoid

g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]

C = [pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]

"""
param = {"kernel": ["sigmoid"],
     "gamma": g,
     "C":C}
random_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
random_search.fit(Xtrain,Ytrain)
clf0 = SVC(kernel=random_search.best_params_["kernel"],gamma=random_search.best_params_["gamma"],C=random_search.best_params_["C"])
print ("Check 1")
print("--- %s seconds ---" % (time.time() - start_time))
print("SVR-Sig------")
"""
################################### SVR-RBF
start_time = time.time()
param= {'gamma': g,
    'kernel': ['rbf'],
    'C': C}
grid_search = RandomizedSearchCV(mod,param,n_jobs=1,n_iter=100)
grid_search.fit(Xtrain,Ytrain)           
clf1 = SVC(gamma = grid_search.best_params_["gamma"],kernel=grid_search.best_params_["kernel"],C=grid_search.best_params_["C"])
clf1.fit(Xtrain,Ytrain)
zpred = clf1.predict(Xtest)

print("Accuracy: ",accuracy(zpred,Ytest)) 
print("--- %s seconds ---" % (time.time() - start_time))


cat_feature = ["Gender","Married","Dependents","Education","Self_Employed","Loan_Amount_Term","Credit_History","Property_Area"]
num_feature = ["ApplicantIncome","CoapplicantIncome","LoanAmount"]

z_out = clf1.predict(X12)
out = []
for num in z_out:
    if num==1:
        out.append("Y")
    else:
        out.append("N")
z = pd.DataFrame({"Loan_ID":t2,"Loan_Status":out})
z.to_csv("C:\\Users\\ndasadhi\\Desktop\\Competitions\\Loan\\output.csv", index=False)