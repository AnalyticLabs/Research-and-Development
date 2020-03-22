"Importing the Packages"
import os
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import specgram
import glob 

#%% External Defined Functions

#def load_sound_files(file_paths):
#    raw_sounds = []
#    for fp in file_paths:
#        X,sr = librosa.load(fp)
#        raw_sounds.append(X)
#    return raw_sounds
#
#def plot_waves(sound_names,raw_sounds):
#    i = 1
#    fig = plt.figure(figsize=(25,60), dpi = 900)
#    for n,f in zip(sound_names,raw_sounds):
#        plt.subplot(10,1,i)
#        librosa.display.waveplot(np.array(f),sr=22050)
#        plt.title(n.title())
#        i += 1
#    plt.suptitle("Figure 1: Waveplot",x=0.5, y=0.915,fontsize=18)
#    plt.show()
#    
#def plot_specgram(sound_names,raw_sounds):
#    i = 1
#    fig = plt.figure(figsize=(25,60), dpi = 900)
#    for n,f in zip(sound_names,raw_sounds):
#        plt.subplot(10,1,i)
#        specgram(np.array(f), Fs=22050)
#        plt.title(n.title())
#        i += 1
#    plt.suptitle("Figure 2: Spectrogram",x=0.5, y=0.915,fontsize=18)
#    plt.show()
#    
#def plot_log_power_specgram(sound_names,raw_sounds):
#    i = 1
#    fig = plt.figure(figsize=(25,60), dpi = 900)
#    for n,f in zip(sound_names,raw_sounds):
#        plt.subplot(10,1,i)
#        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
#        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
#        plt.title(n.title())
#        i += 1
#    plt.suptitle("Figure 3: Log power spectrogram",x=0.5, y=0.915,fontsize=18)
#    plt.show()
#    
#def plot_confusion_matrix(y_true, y_pred, classes,
#                          normalize=False,
#                          title=None,
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if not title:
#        if normalize:
#            title = 'Normalized confusion matrix'
#        else:
#            title = 'Confusion matrix, without normalization'
#
#    # Compute confusion matrix
#    cm = confusion_matrix(y_true, y_pred)
#    # Only use the labels that appear in the data
#    #classes = classes[unique_labels(y_true, y_pred)]
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    fig, ax = plt.subplots()
#    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#    ax.figure.colorbar(im, ax=ax)
#    # We want to show all ticks...
#    ax.set(xticks=np.arange(cm.shape[1]),
#           yticks=np.arange(cm.shape[0]),
#           # ... and label them with the respective list entries
#           xticklabels=classes, yticklabels=classes,
#           title=title,
#           ylabel='True label',
#           xlabel='Predicted label')
#
#    # Rotate the tick labels and set their alignment.
#    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#             rotation_mode="anchor")
#
#    # Loop over data dimensions and create text annotations.
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i in range(cm.shape[0]):
#        for j in range(cm.shape[1]):
#            ax.text(j, i, format(cm[i, j], fmt),
#                    ha="center", va="center",
#                    color="white" if cm[i, j] > thresh else "black")
#    fig.tight_layout()
#    return ax
    
#%% 
"Extracting the Features"
def extract_feature_mfccs_chroma_mel_contrast_tonnetz(signal,fs):
    X=signal
    sample_rate = fs
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) # 40 features
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0) # 12 features
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) # 128 features
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) # 7 features
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0) # 6 features
    return list(mfccs)+list(chroma)+list(mel)+list(contrast)+list(tonnetz) # 193 features


def get_all_wav_files(path, ids):
    files=[]
    for i in ids:
        files+=[os.path.join(path,str(i)+'.wav')]
    return files

#%%
"training data feature extraction and storing in a dataframe"
#train_path = r'D:\Sound classification\Urban sound classification\data\Train'
#train_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_labels.csv')
#
#train_files = get_all_wav_files(train_path, train_labels['ID'])
#
#train_features = []
#for i in range(len(train_files)):
#    print(i)
#    path = train_files[i]
#    signal, sf = librosa.load(path)
#    features = extract_feature_mfccs_chroma_mel_contrast_tonnetz(signal, sf)
#    train_features+=[features]
#
#train_features1 = pd.DataFrame(train_features)
#train_features1.to_csv(r'D:\Sound classification\Urban sound classification\data\train_features_mfccs_chroma_mel_contrast_tonnetz.csv')


"testing data feature extraction and storing in a dataframe"
#test_path = r'D:\Sound classification\Urban sound classification\data\Test'
#test_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_labels.csv')
#
#test_files = get_all_wav_files(test_path, test_labels['ID'])
#
#test_features = []
#for i in range(len(test_files)):
#    if i%10==0:
#        print(i)
#    path = test_files[i]
#    signal, sf = librosa.load(path)
#    features = extract_feature_mfccs_chroma_mel_contrast_tonnetz(signal, sf)
#    test_features+=[features]
#
#test_features1 = pd.DataFrame(test_features)
#test_features1.to_csv(r'D:\Sound classification\Urban sound classification\data\test_features_mfccs_chroma_mel_contrast_tonnetz.csv')

#%%
train_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_features_mfccs_chroma_mel_contrast_tonnetz.csv').drop(columns=['Unnamed: 0'])
train_labels = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\train_labels.csv')
test_features = pd.read_csv(r'D:\Sound classification\Urban sound classification\data\test_features_mfccs_chroma_mel_contrast_tonnetz.csv').drop(columns=['Unnamed: 0'])
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
    feature_types = ['MFCCS','CHROMA','MEL','CONTRAST','TONNETZ','MFCCS_CHROMA_MEL_CONTRAST_TONNETZ']
    feat_dict = {'MFCCS':[0,40],'CHROMA':[40,52],'MEL':[52,180],'CONTRAST':[180,187],'TONNETZ':[187,193],'MFCCS_CHROMA_MEL_CONTRAST_TONNETZ':[0,193]}
    for feat in feature_types:
        st = time.time()
        print(":::::::MODEL::::::",model_dict[model])
        print(":::::FEATURES::::::",feat)
        a = feat_dict[feat][0]
        b = feat_dict[feat][1]
        
        x_train1 = x_train[:,a:b]
        x_val1 = x_val[:,a:b]
        
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
#        x_test1 = x_test[:,a:b]
#        y_pred_test = validation(x_train1,y_train,x_test1, model)    
#        test_labels['Class'] = [le.classes_[i] for i in y_pred_test]
#        test_labels.to_csv(r'D:\Sound classification\Urban sound classification\Analytics Vidhya Submissions\sub_ram_7.csv')

        i+=1
#results_acc.to_csv(r'D:\Sound classification\Urban sound classification\Results\Stratified_5_fold_CV_Accuracy_Val_Accuracy_for_different_models_with_default_parameters(mfccs_chroma_mel_contrsat_tonnetz).csv')
