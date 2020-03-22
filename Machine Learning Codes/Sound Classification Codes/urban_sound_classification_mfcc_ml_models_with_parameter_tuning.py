
import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"ML models"
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

"performace measures"
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

#%%
train_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_features.csv').drop(columns=['Unnamed: 0'])
train_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_labels.csv')
test_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_features.csv').drop(columns=['Unnamed: 0'])
test_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_labels.csv')

le = LabelEncoder()
target = le.fit_transform(train_labels['Class'])


x_train, x_val, y_train, y_val = train_test_split(train_features, target, random_state=10, test_size=0.1)

"validation size is selected as 38%, because actual size is also 38% of (train data + test data)"
#x_train, x_val, y_train, y_val = train_test_split(train_features, target, random_state=10, test_size=0.38)


x_train = pd.DataFrame(x_train).values
y_train = pd.DataFrame(y_train).values
x_val = pd.DataFrame(x_val).values
y_val = pd.DataFrame(y_val).values

        
lr = LogisticRegression() 
gnb  = GaussianNB()
svc = SVC()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
xgbc = XGBClassifier()
knnc = KNeighborsClassifier()

models = [lr, gnb, svc, dtc, rfc, xgbc, knnc]
model_dict = {lr: 'LOGISTIC REGRESSION',
              gnb: 'GAUSSIAN NB',
              svc: 'SUPPORT VECTOR CLASSIFIER',
              dtc: 'DECISION TREE CLASSIFIER',
              rfc: 'RANDOM FOREST CLASSIFIER',
              xgbc: 'XGB CLASSIFIER',
              knnc: 'K-NEAREST NEIGHBORS CLASSIFIER'}

#%%
"Here, we'll be using MFCC features only"
import time
st = time.time()
results_acc = pd.DataFrame(np.zeros((7,5))*np.nan,columns=['Model','Features','Tuned parameters','Tuned parameter_values','VAL_ACC'])
i=0
grid_params_list = {lr:{'penalty' : ['l1', 'l2'],'C' : np.logspace(-4, 4, 20)},
                    gnb:{'var_smoothing':[1e-08,1e-09,1e-10]},
                    svc:{'C': [0.1, 1],'gamma': [1, 0.1],'kernel': ['linear']},
#                    svc:{'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                    dtc:{'criterion':['gini','entropy'],'max_features':['sqrt',None,'log2'],},
                    rfc:{'n_estimators' : [10,50,100,200],'max_features' : ['sqrt',None,'log2']},
#                    rfc:{'n_estimators' : list(range(10,101,10)),'max_features' : list(range(6,32,5))},
                    xgbc:{'eta':[0.1,0.3,0.5,0.7],'gamma':[0,10,100],'max_depth':[1,3,5,9,11]},
                    knnc:{'n_neighbors': [3, 5, 7, 9, 11, 15],'weights': ['uniform', 'distance'],'metric': ['euclidean', 'manhattan']}}
for model in models[6:7]:
#for model in models[1:2]:
    feature_types = ['FFT','PSD','AR','MFCC','FFT_PSD','FFT_AR','FFT_MFCC','PSD_AR','PSD_MFCC','FFT_PSD_AR','FFT_PSD_MFCC','FFT_AR_MFCC','PSD_AR_MFCC','FFT_PSD_AR_MFCC']
    feat_dict = {'FFT':[0,20,20,20],'PSD':[20,40,40,40],'AR':[40,60,60,60],'MFCC':[60,100,100,100],
                 'FFT_PSD':[0,40,40,40],'FFT_AR':[0,20,40,60],'FFT_MFCC':[0,20,60,100],'PSD_AR':[20,60,60,60],'PSD_MFCC':[20,40,60,100],'AR_MFCC':[40,100,100,100],
                 'FFT_PSD_AR':[0,60,60,60],'FFT_PSD_MFCC':[0,40,60,100],'FFT_AR_MFCC':[0,20,40,100],'PSD_AR_MFCC':[20,100,100,100],
                 'FFT_PSD_AR_MFCC':[0,100,100,100]}
    for feat in feature_types[3:4]:
        print(":::::::MODEL::::::",model_dict[model])
        print(":::::FEATURES::::::",feat)
        a = feat_dict[feat][0]
        b = feat_dict[feat][1]
        c = feat_dict[feat][2]
        d = feat_dict[feat][3]
        
        
        x_train1 = np.concatenate((x_train[:,a:b],x_train[:,c:d]), axis=1)
        x_val1 = np.concatenate((x_val[:,a:b],x_val[:,c:d]), axis=1)
        
        final_model = GridSearchCV(model, grid_params_list[model], cv=5, n_jobs=-1)
        final_model.fit(x_train1, y_train)
        y_results = final_model.cv_results_
        y_params = final_model.best_params_
#        print(final_model.cv_results_)
#        print(final_model.best_params_)
        y_pred = final_model.predict(x_val1)

        print('Validation accuracy:',accuracy_score(y_val,y_pred))
        print('Confusion Matrix:',confusion_matrix(y_val, y_pred))
        
#        print("Accuracy (Mean: .%f, Std: %.f)"%(round(np.mean(accuracy_list),3), round(np.std(accuracy_list),5)))    
    
        "storing in a dataframe"
        results_acc['Model'][i] = model_dict[model]
        results_acc['Features'][i] = feat
        results_acc['Tuned parameters'][i] = list(y_params)
        results_acc['Tuned parameter_values'][i] = list(y_params.values())
        results_acc['VAL_ACC'][i] = round(accuracy_score(y_val,y_pred),3)
        
        
        "testing"
        full_train = train_features.values
        full_train1 = np.concatenate((full_train[:,a:b],full_train[:,c:d]), axis=1)
        final_model1 = GridSearchCV(model, grid_params_list[model], cv=5, n_jobs=-1)
        final_model1.fit(full_train1, target)
        full__results = final_model.cv_results_
        full_params = final_model1.best_params_
#        
        x_test = test_features.values
        x_test1 = np.concatenate((x_test[:,a:b],x_test[:,c:d]), axis=1)
        y_pred_test = final_model1.predict(x_test1)    
        test_labels['Class'] = [le.classes_[i] for i in y_pred_test]
        test_labels.to_csv(r'D:\Sound classification\Urban sound classification\Analytics Vidhya Submissions\sub_ram_knn_tuned_mfcc_.62_training_5.csv')

        i+=1
        print(str(time.time()-st)+' seconds')
#results_acc.to_csv(r'D:\Sound classification\Urban sound classification\Results\Stratified_5_fold_CV_Accuracy_Val_Accuracy_for_different_models_with_default_parameters11.csv')
