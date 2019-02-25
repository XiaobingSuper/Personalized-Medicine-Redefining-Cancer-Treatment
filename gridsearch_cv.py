from sklearn import *
import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics,preprocessing  #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


train = np.load('new_train.npy')
test = np.load('new_test.npy')
y = np.load('label.npy')
pid = np.load('pid.npy')

train, x_test, y, y_test = model_selection.train_test_split(train, y, test_size=0, random_state=33)


xgb1 = XGBClassifier(learning_rate =0.05,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,
 colsample_bytree=0.8,objective= 'multi:softprob',scale_pos_weight=1,seed=123)
xgb_param = xgb1.get_xgb_params()
xgb_param['num_class']=9
xgb_param['eval_metric']='mlogloss'
print xgb_param

model = xgb.cv(xgb_param, xgb.DMatrix(train, y), 1000, verbose_eval=50, early_stopping_rounds=100,nfold=5)

print model

param_test1 = {
 'max_depth':[5],
 'min_child_weight':[1]}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.05, n_estimators=160, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
 objective= 'multi:softprob',scale_pos_weight=1, seed=123), 
 param_grid = param_test1,     scoring='neg_log_loss',iid=False, cv=5)
gsearch1.fit(train,y)
print gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_

param_test2 = {
 'max_depth':[4,5,6],
 }
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.05, n_estimators=150, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
 objective= 'multi:softprob',scale_pos_weight=1, seed=123), 
 param_grid = param_test2,     scoring='neg_log_loss',iid=False, cv=5)
gsearch2.fit(train,y)
print gsearch2.grid_scores_, gsearch2.best_params_,     gsearch2.best_score_


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)],
 }
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.05, n_estimators=150, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
 objective= 'multi:softprob',scale_pos_weight=1, seed=123), 
 param_grid = param_test3,     scoring='neg_log_loss',iid=False, cv=5)
gsearch3.fit(train,y)
print gsearch3.grid_scores_, gsearch3.best_params_,     gsearch3.best_score_

param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
 }
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.05, n_estimators=150, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,
 objective= 'multi:softprob',scale_pos_weight=1, seed=123), 
 param_grid = param_test4,     scoring='neg_log_loss',iid=False, cv=5)
gsearch4.fit(train,y)
print gsearch4.grid_scores_, gsearch4.best_params_,     gsearch4.best_score_




