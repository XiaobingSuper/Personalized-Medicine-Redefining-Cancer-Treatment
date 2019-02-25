from sklearn import preprocessing, pipeline, feature_extraction, decomposition, model_selection, metrics, cross_validation, svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize, Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost.sklearn import XGBClassifier
import sklearn
import pandas as pd
import numpy as np
import xgboost as xgb


train = np.load('new_train.npy')
test = np.load('new_test.npy')
y = np.load('label.npy')
pid = np.load('pid.npy')

x1, x_test, y1, y_test = model_selection.train_test_split(train, y, test_size=0, random_state=33)

xgb1 = XGBClassifier(learning_rate =0.01,n_estimators=2000,max_depth=6,min_child_weight=4,gamma=0,subsample=0.7,
 colsample_bytree=0.8,objective= 'multi:softprob',scale_pos_weight=1,seed=0)
xgb_param = xgb1.get_xgb_params()
xgb_param['num_class']=9
xgb_param['eval_metric']='mlogloss'
print xgb_param

#model = xgb.cv(xgb_param, xgb.DMatrix(x1,y1), 2000, verbose_eval=50, early_stopping_rounds=100,nfold=5)

#print model



denom = 0
fold = 1
for i in range(fold):
    
    #x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.15, random_state=i)
    #x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.2, random_state=i,stratify = y)
    #watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    #model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
    #print model.best_ntree_limit
    model = xgb.train(xgb_param, xgb.DMatrix(x1, y1), 200 )
    score1 = metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
    print(score1)
    #if score < 0.9:
    if denom != 0:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        preds += pred
    else:
        pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
        preds = pred.copy()
    denom += 1
    submission = pd.DataFrame(pred, columns=['class'+str(c+1) for c in range(9)])
    submission['ID'] = pid
    submission.to_csv('submission_xgb_origin_day3_1_sub_'+str(i)+'.csv', index=False)
preds /= float(denom)
submission = pd.DataFrame(preds, columns=['class'+str(c+1) for c in range(9)])
submission['ID'] = pid
submission.to_csv('submission_xgb_origin_day3_1.csv', index=False)


