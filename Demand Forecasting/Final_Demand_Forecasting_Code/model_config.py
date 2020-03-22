# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:35:54 2019

@author: AKE9KOR
"""

#%%
#sklearn libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#%%
def models_ML(cluster):
    models = dict()
    n_trees = 100
#    random_state=42
    #prameters for RandomSearch
    lr_param = {"fit_intercept": [True, False],"normalize": [False],"copy_X": [True, False]}
    knn_param = {"n_neighbors":[2,3,4,5,6,7,8],"metric": ["euclidean", "cityblock"]}
    dtree_param = {"max_depth": [3,None],"min_samples_leaf": sp_randint(1, 11),"criterion": ["mse"],"splitter": ["best","random"],"max_features": ["auto","sqrt",None]}
    lasso_param = {"alpha":[0.02, 0.024, 0.025, 0.026, 0.03],"fit_intercept": [True, False],"normalize": [True, False],"selection": ["random"]}
    ridge_param = {"alpha":[200, 230, 250,265, 270, 275, 290, 300, 500],"fit_intercept": [True, False],"normalize": [True, False],"solver": ["auto"]}
    elas_param = {"alpha": list(np.logspace(-5,2,8)),"l1_ratio": [.2,.4,.6,.8],"fit_intercept": [True, False],"normalize": [True,False],"precompute": [True, False]}
#    g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#    c=[pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
#    svr_param={"C":c,"gamma":g,"kernel":["rbf","sigmoid"]}
#    gb_param={"n_estimators":[1, 2, 4, 8, 16, 32, 64, 100, 200],"max_depths":list(np.linspace(1, 32, 32, endpoint=True)),"min_samples_splits":list(np.linspace(0.1, 1.0, 10, endpoint=True)),"min_samples_leaf":list(np.linspace(0.1,0.5,5, endpoint=True)),"max_features":list(range(1,5))}



    if cluster in[1,4,7,10,13,16,19,22,25]:
        #Highly sparse data tree based algorithms
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
        models['bag']               = BaggingRegressor(n_estimators=n_trees)
        models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)
    elif cluster==2:
        models['llars']             = LassoLars()
        models['knn']               = KNeighborsRegressor(n_neighbors=7)
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
        models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==3:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
        models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==5:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['huber']             = HuberRegressor()
        models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['svmr']              = SVR()
    elif cluster==6:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['huber']             = HuberRegressor()
        models['svmr']              = SVR()
        models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==8:
        models['llars']             = LassoLars()
        models['svmr']              = SVR()
        models['lr']               = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['huber']             = HuberRegressor()
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==9:
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100)
        models['bag']               = BaggingRegressor(n_estimators=n_trees)
        models['rf']                = RandomForestRegressor(n_estimators=n_trees)
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees)
    elif cluster==11:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['en']                = RandomizedSearchCV(ElasticNet(), elas_param,scoring='neg_mean_squared_error',n_jobs=1, n_iter= 100,cv=10,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==12:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
        models['knn']               = KNeighborsRegressor(n_neighbors=3)
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)
    elif cluster==14:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100)
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100)
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)
    elif cluster==15:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['en']                = RandomizedSearchCV(ElasticNet(), elas_param,scoring='neg_mean_squared_error', n_jobs=1, n_iter= 100,cv=10,random_state=42)
        models['huber']             = HuberRegressor()
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==17:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['bag']               = BaggingRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==18:
        models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
        models['ridge']             = RandomizedSearchCV(Ridge(), ridge_param, n_jobs=1, n_iter = 100,random_state=42)
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
    elif cluster==20:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['huber']             = HuberRegressor()
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
        models['bag']               = BaggingRegressor(n_estimators=n_trees)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
        models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==21:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
        models['svmr']              = SVR()
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)
    elif cluster==23:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['huber']             = HuberRegressor()
        models['bag']               = BaggingRegressor(n_estimators=n_trees)
        models['svmr']              = SVR()
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==24:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
        models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)
    elif cluster==26:
        models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
        models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
        models['en']                = RandomizedSearchCV(ElasticNet(), elas_param,scoring='neg_mean_squared_error', n_jobs=1, n_iter= 100,cv=10,random_state=42)
        models['extra']             = ExtraTreeRegressor(random_state=42)
        models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    elif cluster==27:
        models['svmr']              = SVR()
        models['knn']               = KNeighborsRegressor(n_neighbors=3)
        models['bag']               = BaggingRegressor(n_estimators=n_trees)
        models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
        models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)


    return models
#%%
def init_ML_hyp_models():
    models = dict()
    n_trees = 100
    #prameters for RandomSearch
    lr_param = {"fit_intercept": [True, False],"normalize": [False],"copy_X": [True, False]}
    knn_param = {"n_neighbors":[2,3,4,5,6,7,8],"metric": ["euclidean", "cityblock"]}
    dtree_param = {"max_depth": [3,None],"min_samples_leaf": sp_randint(1, 11),"criterion": ["mse"],"splitter": ["best","random"],"max_features": ["auto","sqrt",None]}
    lasso_param = {"alpha":[0.02, 0.024, 0.025, 0.026, 0.03],"fit_intercept": [True, False],"normalize": [True, False],"selection": ["random"]}
    ridge_param = {"alpha":[200, 230, 250,265, 270, 275, 290, 300, 500],"fit_intercept": [True, False],"normalize": [True, False],"solver": ["auto"]}
    elas_param = {"alpha": list(np.logspace(-5,2,8)),"l1_ratio": [.2,.4,.6,.8],"fit_intercept": [True, False],"normalize": [True,False],"precompute": [True, False]}
#    g = [pow(2,-15),pow(2,-14),pow(2,-13),pow(2,-12),pow(2,-11),pow(2,-10),pow(2,-9),pow(2,-8),pow(2,-7),pow(2,-6),pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3)]
#    c=[pow(2,-5),pow(2,-4),pow(2,-3),pow(2,-2),pow(2,-1),pow(1,0),pow(2,1),pow(2,2),pow(2,3),pow(2,4),pow(2,5),pow(2,6),pow(2,7),pow(2,8),pow(2,9),pow(2,10),pow(2,11),pow(2,12),pow(2,13),pow(2,14),pow(2,15)]
#    svr_param={"C":c,"gamma":g,"kernel":["rbf","sigmoid"]}
#    gb_param={"n_estimators":[1, 2, 4, 8, 16, 32, 64, 100, 200],"max_depths":list(np.linspace(1, 32, 32, endpoint=True)),"min_samples_splits":list(np.linspace(0.1, 1.0, 10, endpoint=True)),"min_samples_leaf":list(np.linspace(0.1,0.5,5, endpoint=True)),"max_features":list(range(1,5))}
    models['lr']                = RandomizedSearchCV(LinearRegression(), lr_param, n_jobs = 1,random_state=42)
    models['lasso']             = RandomizedSearchCV(Lasso(), lasso_param, n_jobs=1, n_iter = 100,random_state=42)
    models['ridge']             = RandomizedSearchCV(Ridge(), ridge_param, n_jobs=1, n_iter = 100,random_state=42)
    models['en']                = RandomizedSearchCV(ElasticNet(), elas_param,scoring='neg_mean_squared_error', n_jobs=1, n_iter= 100,cv=10,random_state=42)
    models['huber']             = HuberRegressor()
    models['llars']             = LassoLars()
    models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3,random_state=42)
    models['knn']               = KNeighborsRegressor(n_neighbors=3)
    models['cart']              = RandomizedSearchCV(DecisionTreeRegressor(),dtree_param,n_jobs=1,n_iter=100,random_state=42)
    models['extra']             = ExtraTreeRegressor(random_state=42)
    models['svmr']              = SVR()

    n_trees = 100
    models['ada']               = AdaBoostRegressor(n_estimators=n_trees,random_state=42)
    models['bag']               = BaggingRegressor(n_estimators=n_trees)
    models['rf']                = RandomForestRegressor(n_estimators=n_trees,random_state=42)
    models['et']                = ExtraTreesRegressor(n_estimators=n_trees,random_state=42)
    models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees,random_state=42)

    return models

#%%
def init_ML_models():
    models = dict()
    models['lr']                = LinearRegression()
    models['lasso']             = Lasso()
    models['ridge']             = Ridge()
    models['en']                = ElasticNet()
    models['huber']             = HuberRegressor()
    models['llars']             = LassoLars()
    models['pa']                = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
#    models['knn']               = KNeighborsRegressor(n_neighbors=5)
    models['cart']              = DecisionTreeRegressor()
    models['extra']             = ExtraTreeRegressor()
    models['svmr']              = SVR()

    n_trees = 100
    models['ada']               = AdaBoostRegressor(n_estimators=n_trees)
    models['bag']               = BaggingRegressor(n_estimators=n_trees)
    models['rf']                = RandomForestRegressor(n_estimators=n_trees)
    models['et']                = ExtraTreesRegressor(n_estimators=n_trees)
    models['gbm']               = GradientBoostingRegressor(n_estimators=n_trees)

    return models

#%%
def models_arima(order,cluster):
    models = dict()
    if cluster in [2,3,5,6,8,9,20,21,23,24,26,27]:
        #Except for clusters with high sparsity and Non-Seasonal data
        models['ARIMA']              = order
    elif cluster in [1,4,7,10,13,16,19,22,25]:
        #For highly sparse clusters
        models['AR']                 = (order[0], 0, 0)
        models['MA']                 = (0, 0, order[2])
        models['ARMA']               = (order[0],order[2])
    return models

#%%
def init_ARIMA_models(order):
    models = dict()
    models['AR']                 = (order[0], 0, 0)
    models['MA']                 = (0, 0, order[2])
    models['ARMA']               = (order[0],order[2])
    models['ARIMA']              = order
    return models

#%%
def models_ES(cluster):
    key=[]
#    key=['SES','HWES']
    if cluster in [12,15,18,21,24,27]:
        key=['HWES']#SES
    elif cluster in [2,5,8,11,14,17,20,23,26]:
        key=['HWES']#SES&HWES
    return key
#%%
def init_ES_models():
    models = dict()

    models['SES']               = SimpleExpSmoothing()
    models['HWES']              = ExponentialSmoothing()
    return models
#%%
def models_Averages(cluster):
    key=[]
#    key=['sma','wma']
    if cluster in [11,14,17]:
        key=['sma','wma']
    elif cluster in [2,5,8,20,23,26]:
        key=['wma']
    return key

#%%
def init_test_shape():
    test_shape_incr = dict()
    test_shape_incr['svmr']     = 0
    test_shape_incr['ada']      = 0
    test_shape_incr['lr']       = 0
    test_shape_incr['lasso']    = 0
    test_shape_incr['ridge']    = 0#1
    test_shape_incr['en']       = 0
    test_shape_incr['huber']    = 0#1
    test_shape_incr['llars']    = 0
    test_shape_incr['pa']       = 0
    test_shape_incr['knn']      = 0
    test_shape_incr['cart']     = 0
    test_shape_incr['extra']    = 3
    test_shape_incr['bag']      = 3#1
    test_shape_incr['rf']       = 3#1
    test_shape_incr['et']       = 3#1
    test_shape_incr['gbm']      = 3#1#

    return test_shape_incr
#%%
def test_shape_adder(key):
    if key =='knn':
        test_shape_add=0
    elif key == 'lasso':
        test_shape_add=0
    elif key == 'lr':
        test_shape_add=0
    elif key == 'ridge' or key == 'en' or key == 'huber' or key == 'llars' or key == 'pa':
        test_shape_add=0
    elif key == 'cart' or key == 'extra' or key == 'svmr' or key == 'ada':
        test_shape_add=2
    elif key == 'bag' or key == 'rf' or key == 'et' or key == 'gbm':
         test_shape_add=3
    else:
         test_shape_add=0

    return test_shape_add