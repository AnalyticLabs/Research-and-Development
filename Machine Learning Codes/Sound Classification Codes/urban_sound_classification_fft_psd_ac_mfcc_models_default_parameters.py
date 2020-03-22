
import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt

#%%
"Frequency & time related features"
"FFT"
from scipy.fftpack import fft
def get_fft_values(y_value, N, T):
    f_values = np.linspace(0,1/(2*T),N//2)
    fft_values = 2/N*np.abs(fft(y_value)[0:N//2])
    return f_values, fft_values

"PSD"
from scipy.signal import welch
def get_psd_values(y_value, fs):
    f_values, psd_values = welch(y_value, fs)
    return f_values, psd_values

"Auto-correlation"
def auto_corr(x):
    result = np.correlate(x,x,mode='full')
    return result[len(result)//2:]

def get_autocorr_values(y_value,N,T):
    t_values = np.linspace(0, T*N, N)
    auto_corrvalues = auto_corr(y_value)
    return t_values, auto_corrvalues

"Feature Extraction"
npeaks=10
def get_first_n_peaks(data, no_peaks=npeaks):
    data = data.sort_values(['y'], ascending=False).reset_index(drop=['index'])
    x, y = list(data['x'])+[0]*no_peaks, list(data['y'])+[0]*no_peaks
    x1, y1 = x[:no_peaks],y[:no_peaks]
    return list(x1),list(y1)

def get_features(x_values, y_values):
    indices_peaks = argrelextrema(y_values, np.greater)[0]
    data = pd.DataFrame()
    data['x'] = x_values[indices_peaks]
    data['y'] = y_values[indices_peaks]
    peaks_x, peaks_y = get_first_n_peaks(data)
    
    return peaks_x + peaks_y

def get_mfcc_values(signal, fs):
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=40).T,axis=0)
    return list(mfccs)

def extract_features(signal, fs):
    T = 1 / fs
    N = signal.shape[0]
    #list_of_features = []
    #list_of_labels = []
    features = []
    features += get_features(get_fft_values(signal, N, T)[0],get_fft_values(signal, N, T)[1])
    features += get_features(get_psd_values(signal, fs)[0],get_psd_values(signal, fs)[1])
    features += get_features(get_autocorr_values(signal, N, T)[0],get_autocorr_values(signal, N, T)[1])
    features += get_mfcc_values(signal, fs)
    return features


def get_all_wav_files(path, ids):
    files=[]
    for i in ids:
        files+=[os.path.join(path,str(i)+'.wav')]
    return files
#%%
"training data feature extraction and storing in a dataframe"
#train_path = r'D:\Sound classification\Urban sound classification\data\Train'
#
#train_files = get_all_wav_files(train_path, train_labels['ID'])
#
#train_features = []
#for i in range(len(train_files)):
#    print(i)
#    path = train_files[i]
#    signal, sf = librosa.load(path)
#    features = extract_features(signal, sf)
#    train_features+=[features]
#
#train_features1 = pd.DataFrame(train_features)
##train_features1.to_csv(r'D:\Sound classification\Urban sound classification\data\train_features.csv')
#
#
"testing data feature extraction and storing in a dataframe"
#test_path = r'D:\Sound classification\Urban sound classification\data\Test'
#
#test_files = get_all_wav_files(test_path, test_labels['ID'])
#
#test_features = []
#for i in range(len(test_files)):
#    if i%10==0:
#        print(i)
#    path = test_files[i]
#    signal, sf = librosa.load(path)
#    features = extract_features(signal, sf)
#    test_features+=[features]
#
#test_features1 = pd.DataFrame(test_features)
#test_features1.to_csv(r'D:\Sound classification\Urban sound classification\data\test_features.csv')

#%%
train_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_features.csv').drop(columns=['Unnamed: 0'])
train_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_labels.csv')
test_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_features.csv').drop(columns=['Unnamed: 0'])
test_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_labels.csv')

le = LabelEncoder()
target = le.fit_transform(train_labels['Class'])

x_train, x_val, y_train, y_val = train_test_split(train_features, target, random_state=10, test_size=0.1)

"ML models"
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

"10-fold stratified cross validation"
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

x_train = pd.DataFrame(x_train).values
y_train = pd.DataFrame(y_train).values
x_val = pd.DataFrame(x_val).values
y_val = pd.DataFrame(y_val).values


def stratified_cross_validation(x_train,y_train,model, n_splits1):
    stf = StratifiedKFold(n_splits=n_splits1, shuffle=True, random_state=10)
    accuracy_list = []
    k=0
    for train_index, test_index in stf.split(x_train, y_train):
#        print(k)
        X_TRAIN, X_TEST = x_train[train_index], x_train[test_index]
        Y_TRAIN, Y_TEST = y_train[train_index], y_train[test_index]
        model.fit(X_TRAIN,Y_TRAIN)
        Y_PRED = model.predict(X_TEST)
        accuracy_list+=[round(accuracy_score(Y_TEST, Y_PRED),3)]
        k+=1
    return accuracy_list
        
lr = LogisticRegression() 
gnb  = GaussianNB()
#mnb = MultinomialNB() # tried ut not useful in our case as it requires positive inputs
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

def validation(x_t,y_t,x_v, model):
    model.fit(x_t,y_t)
    y_pred = model.predict(x_v)
    return y_pred
 
def testing(x_tr,y_tr,x_te, model):
    model.fit(x_tr,y_tr)
    y_pred = model.predict(x_te)
    return y_pred
 
#%%
import time

stf_splits = 5
results_acc = pd.DataFrame(np.zeros((98,11)),columns=['Model','Features','sfold-1','sfold-2','sfold-3','sfold-4','sfold-5','Mean(STF_ACC)','Std_dev (STF_ACC)','VAL_ACC','Total Exec. Time (s)'])
i=0
for model in models:
    feature_types = ['FFT','PSD','AR','MFCC','FFT_PSD','FFT_AR','FFT_MFCC','PSD_AR','PSD_MFCC','FFT_PSD_AR','FFT_PSD_MFCC','FFT_AR_MFCC','PSD_AR_MFCC','FFT_PSD_AR_MFCC']
    feat_dict = {'FFT':[0,20,20,20],'PSD':[20,40,40,40],'AR':[40,60,60,60],'MFCC':[60,100,100,100],
                 'FFT_PSD':[0,40,40,40],'FFT_AR':[0,20,40,60],'FFT_MFCC':[0,20,60,100],'PSD_AR':[20,60,60,60],'PSD_MFCC':[20,40,60,100],'AR_MFCC':[40,100,100,100],
                 'FFT_PSD_AR':[0,60,60,60],'FFT_PSD_MFCC':[0,40,60,100],'FFT_AR_MFCC':[0,20,40,100],'PSD_AR_MFCC':[20,100,100,100],
                 'FFT_PSD_AR_MFCC':[0,100,100,100]}
    for feat in feature_types:
        st = time.time()
        print(":::::::MODEL::::::",model_dict[model])
        print(":::::FEATURES::::::",feat)
        a = feat_dict[feat][0]
        b = feat_dict[feat][1]
        c = feat_dict[feat][2]
        d = feat_dict[feat][3]
        
        x_train1 = np.concatenate((x_train[:,a:b],x_train[:,c:d]), axis=1)
        x_val1 = np.concatenate((x_val[:,a:b],x_val[:,c:d]), axis=1)
        
        accuracy_list = stratified_cross_validation(x_train1, y_train, model, stf_splits)
        print("Accuracy_list::", accuracy_list)
        print("Accuracy (Mean: .%f, Std: %.f)"%(round(np.mean(accuracy_list),3), round(np.std(accuracy_list),5)))    
    
        "storing in a dataframe"
        results_acc['Model'][i] = model_dict[model]
        results_acc['Features'][i] = feat
        results_acc.iloc[i,2:7] = accuracy_list
        results_acc['Mean(STF_ACC)'][i] = round(np.mean(accuracy_list),3)
        results_acc['Std_dev (STF_ACC)'][i] = round(np.std(accuracy_list),5)
        
        "validation"
        y_pred_val = validation(x_train1,y_train,x_val1, model)
        
        val_accuracy = round(accuracy_score(y_val, y_pred_val),3)
        print("Validation Accuracy::", val_accuracy)
        
        results_acc['VAL_ACC'][i] = val_accuracy
        results_acc['Total Exec. Time (s)'][i] = round(time.time()-st,2)
        
        "testing"
#        x_test = test_features.values
#        x_test1 = np.concatenate((x_test[:,a:b],x_test[:,c:d]), axis=1)
#        y_pred_test = validation(x_train1,y_train,x_test1, model)    
#        test_labels['Class'] = [le.classes_[i] for i in y_pred_test]
#        test_labels.to_csv(r'D:\Sound classification\Urban sound classification\Analytics Vidhya Submissions\sub_ram_4.csv')

        i+=1
results_acc.to_csv(r'D:\Sound classification\Urban sound classification\Results\Stratified_5_fold_CV_Accuracy_Val_Accuracy_for_different_models_with_default_parameters11.csv')
