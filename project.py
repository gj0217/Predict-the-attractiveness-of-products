from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np, sys
import pandas as pd
import collections
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
import re
import xgboost as xgb
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_words(file):
    with open (file) as f:
        words_box=[]
        for line in f:
            words_box.extend(line.strip().split())
    return collections.Counter(words_box)

def mySub(m):
    m = re.sub(r'&|:|<li>|</li>|ul>|</ul>|\+|\-|\/|\(|\)|\+|\=|\.|%|;|\*|,|-|[0-9]', '', m)
    m = re.sub(r'\s+\d\s+|\s+[a-zA-Z]\s+|\s+([A-Z])+\s', '', m)
    m = re.sub(r'\b(For|for|With|with|and|to|in|of|Set|amp|the|One|The|[a-z]|[A-Z]|In|Intl|And|intl|not|no|has|both|there|as|As|at|GB|Leather|Mi)\b','', m)
    return m

one=get_words('/Users/jing/Desktop/dict_one.txt')
zero=get_words('/Users/jing/Desktop/dict_zero.txt')

one_des=get_words('/Users/jing/Desktop/dict_one_des.txt')
zero_des=get_words('/Users/jing/Desktop/dict_zero_des.txt')

keys_one=[]
keys_zero=[]
keys_zero_des=[]
keys_one_des=[]

for each in one:
  if one[each]>50:
   keys_one.append(each)
for each in zero:
  if zero[each]>50:
   keys_zero.append(each)

for each in one:
  if one_des[each]>200:
   keys_one_des.append(each)
for each in zero:
  if zero_des[each]>200:
   keys_zero_des.append(each)


frequency1=list(set(keys_one).difference(set(keys_zero)))
frequency2=list(set(keys_zero).difference(set(keys_one)))

frequency = list(set(keys_zero).union(set(keys_one)))

frequency1_des=list(set(keys_one_des).difference(set(keys_zero_des)))
frequency2_des=list(set(keys_zero_des).difference(set(keys_one_des)))

frequency_des = list(set(keys_zero_des).union(set(keys_one_des)))

#frequency = frequency1+frequency2

#=] print(frequency)

#pd.set_option('display.max_rows', None)
pd.set_option('display.width',None)

df = pd.read_csv('train_data.csv', encoding='utf-8',usecols=[1,2,3,4,5,6,7])
X=pd.read_csv('train_data.csv')
y=pd.read_csv('train_label.csv')['score']
df = pd.concat([X,y],axis=1)
# df.drop_duplicates(keep='first', inplace=True)  # 去重，只保留第一次出现的样本
#df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.4 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
#print (df.shape, df_test.shape, df_train.shape)
# load train data
X_csv = pd.read_csv('train_data.csv',usecols=[1,2,3,4,5,6,7])


#X_csv=pd.concat([df_train['name'],df_train['lvl1'],df_train['lvl2'],df_train['lvl3'],df_train['descrption'],df_train['price'],df_train['type']],axis=1)

# load train label
y_train = pd.read_csv('train_label.csv')['score']
#y_train=df_train['score']
# combine X_train & y_train
temp=pd.concat([X_csv,y_train],axis=1)
# delete duplicated rows
#temp.drop_duplicates(inplace=True)
temp=temp[temp['price']>0]
# delete rows containing NaN
#temp=temp.fillna(-999)

#temp=temp['lvl3'].fillna('Phone cases')




temp['quality']=temp['descrption'].str.extract('(high-quality|quality)', expand=False)
temp['high']=temp['descrption'].str.extract('(high)', expand=False)
temp['100%']=temp['descrption'].str.extract('(100%)', expand=False)
temp['fashion']=temp['descrption'].str.extract('(fashion)', expand=False)
temp['comfortable']=temp['descrption'].str.extract('(comfortable|comfort)', expand=False)
temp['durable']=temp['descrption'].str.extract('(durable)', expand=False)
# temp['iphone']=temp['name'].str.extract('(iphone|phone)', expand=False)
# temp['usb']=temp['name'].str.extract('(usb)', expand=False)
# temp['case']=temp['name'].str.extract('(case)', expand=False)
# temp['black']=temp['name'].str.extract('(black)', expand=False)
# temp['cover']=temp['name'].str.extract('(cover)', expand=False)
# temp['Samsung']=temp['name'].str.extract('(Samsung|Galaxy)', expand=False)
# temp['watch']=temp['name'].str.extract('(watch)', expand=False)
# temp['human']=temp['name'].str.extract('(woman|man)', expand=False)
frequency_final=[]
for i in frequency:
    frequency[frequency.index(i)]=mySub(i)
    i=mySub(i)
    if (i.strip()!=''):
        word_ex = '('+i+')'
        temp[i] = temp['name'].str.extract(word_ex, expand=False)
        temp[i].fillna('no_'+i,inplace = True)
        frequency_final.append(i)

#print(frequency_final)
frequency_des_final=[]
# for i in frequency_des:
#     frequency_des[frequency_des.index(i)]=mySub(i)
#     i=mySub(i)
#     if (i.strip()!=''):
#         word_ex = '('+i+')'
#         # print(word_ex)
#         temp[i] = temp['descrption'].str.extract(word_ex, expand=False)
#         temp[i].fillna('no_'+i,inplace = True)
#         frequency_des_final.append(i)

# for i in frequency_des:
#     print('frequency_des:',i)
#
# for i in frequency:
#     print('frequency:',i)

#print(frequency_des_final)

temp['quality'].fillna('no_quality',inplace = True)
temp['high'].fillna('no_high',inplace = True)
temp['100%'].fillna('no_100%',inplace = True)
temp['fashion'].fillna('no_fashion',inplace = True)
temp['comfortable'].fillna('no_comfortable',inplace = True)
temp['durable'].fillna('no_durable',inplace = True)
# temp['iphone'].fillna('no_iphone',inplace = True)
# temp['usb'].fillna('no_usb',inplace = True)
# temp['case'].fillna('no_case',inplace = True)
# temp['black'].fillna('no_black',inplace = True)
# temp['cover'].fillna('no_cover',inplace = True)
# temp['Samsung'].fillna('no_Samsung',inplace = True)
# temp['watch'].fillna('no_watch',inplace = True)
# temp['human'].fillna('no_woman',inplace = True)

# train_ext0=temp[temp.columns[14:15]]
# print(train_ext0[train_ext0.isnull().values==False])
print(temp.dtypes)
#temp_dum=pd.get_dummies(temp,columns=['iphone','lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable','usb','case','black','cover','Samsung','watch','human'],dummy_na=True)
# temp_dum=pd.get_dummies(temp,columns=['lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable'],dummy_na=True)
temp_dum=pd.get_dummies(temp,columns=['lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable']+frequency_final,dummy_na=True)
#temp_dum=pd.get_dummies(temp,columns=['lvl1','lvl2','lvl3','type','100%']+frequency_final,dummy_na=True)

X_train=temp_dum.drop(['name','score','descrption'],axis=1)
y_train = temp_dum['score']

# X_train=temp.drop(['name','score','descrption'],axis=1)
# y_train = temp['score']

X_csv_test = pd.read_csv('test_data.csv',usecols=[1,2,3,4,5,6,7])
#X_csv_test = pd.concat([df_test['name'],df_test['lvl1'],df_test['lvl2'],df_test['lvl3'],df_test['descrption'],df_test['price'],df_test['type']],axis=1)
#
#X_csv_test=X_csv_test.fillna(-999)
#print(X_train['descrption'])
X_csv_test['quality']=X_csv_test['descrption'].str.extract('(high-quality|quality)', expand=False)
X_csv_test['high']=X_csv_test['descrption'].str.extract('(high)', expand=False)
X_csv_test['100%']=X_csv_test['descrption'].str.extract('(100%)', expand=False)
X_csv_test['fashion']=X_csv_test['descrption'].str.extract('(fashion)', expand=False)
X_csv_test['comfortable']=X_csv_test['descrption'].str.extract('(comfortable|comfort)', expand=False)
X_csv_test['durable']=X_csv_test['descrption'].str.extract('(durable)', expand=False)
# X_csv_test['iphone']=X_csv_test['name'].str.extract('(iphone|phone)', expand=False)
# X_csv_test['usb']=X_csv_test['name'].str.extract('(usb)', expand=False)
# X_csv_test['case']=X_csv_test['name'].str.extract('(case)', expand=False)
# X_csv_test['black']=X_csv_test['name'].str.extract('(black)', expand=False)
# X_csv_test['cover']=X_csv_test['name'].str.extract('(cover)', expand=False)
# X_csv_test['Samsung']=X_csv_test['name'].str.extract('(Samsung|Galaxy)', expand=False)
# X_csv_test['watch']=X_csv_test['name'].str.extract('(watch)', expand=False)
# X_csv_test['human']=X_csv_test['name'].str.extract('(woman|man)', expand=False)

for i in frequency_final:
    # frequency[frequency.index(i)] = mySub(i)
    # i=mySub(i)
    # if (i.strip()!=''):
        word_ex='('+i+')'
        X_csv_test[i] = X_csv_test['name'].str.extract(word_ex, expand=False)
        X_csv_test[i].fillna('no_'+i,inplace = True)

# for i in frequency_des_final:
#     # frequency_des[frequency_des.index(i)] = mySub(i)
#     # i=mySub(i)
#     # if (i.strip()!=''):
#         word_ex='('+i+')'
#         X_csv_test[i] = X_csv_test['descrption'].str.extract(word_ex, expand=False)
#         X_csv_test[i].fillna('no_'+i,inplace = True)

# for i in frequency:
#     print(i)

X_csv_test['quality'].fillna('no_quality',inplace = True)
X_csv_test['high'].fillna('no_high',inplace = True)
X_csv_test['100%'].fillna('no_100%',inplace = True)
X_csv_test['fashion'].fillna('no_fashion',inplace = True)
X_csv_test['comfortable'].fillna('no_comfortable',inplace = True)
X_csv_test['durable'].fillna('no_durable',inplace = True)
# X_csv_test['iphone'].fillna('no_iphone',inplace = True)
# X_csv_test['usb'].fillna('no_usb',inplace = True)
# X_csv_test['case'].fillna('no_case',inplace = True)
# X_csv_test['black'].fillna('no_black',inplace = True)
# X_csv_test['cover'].fillna('no_cover',inplace = True)
# X_csv_test['Samsung'].fillna('no_Samsung',inplace = True)
# X_csv_test['watch'].fillna('no_watch',inplace = True)
# X_csv_test['human'].fillna('no_human',inplace = True)

X_csv_test.drop(['name','descrption'],axis=1,inplace=True)

# temp_dum_test=pd.get_dummies(X_csv_test,columns=['iphone','lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable','usb','case','black','cover','Samsung','watch','human'],dummy_na=True)
# temp_dum_test=pd.get_dummies(X_csv_test,columns=['lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable'],dummy_na=True)
temp_dum_test=pd.get_dummies(X_csv_test,columns=['lvl1','lvl2','lvl3','type','quality','high','100%','fashion','comfortable','durable']+frequency_final,dummy_na=True)
#temp_dum_test=pd.get_dummies(X_csv_test,columns=['lvl1','lvl2','lvl3','type','100%']+frequency_final,dummy_na=True)
X_train=MinMaxScaler().fit_transform(X_train)
X_test=MinMaxScaler().fit_transform(temp_dum_test)

# X_test=X_csv_test
# categorical_features_indices = np.where(X_train.dtypes != np.float)[0]



#X_test=temp_dum_test


# rfr = RandomForestRegressor(n_estimators= 300, max_depth=50, min_samples_split=55,
#                                  min_samples_leaf=12,max_features=72,oob_score=True, random_state=100)
# rfr.fit(X_train,y_train)
# rfr_y_predict=rfr.predict(X_test)
#
# f, ax = plt.subplots(figsize=(7, 5))
# ax.bar(range(len(rfr.feature_importances_)),rfr.feature_importances_)
# ax.set_title("Feature Importances")
# f.show()



y_true=df_test['score']

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'metric': 'binary_logloss',
    'max_depth':50,
    'random_state':10
}

cat_model = CatBoostRegressor(iterations=700,
                             learning_rate=0.02,
                             depth=6,
                             eval_metric='RMSE',
                             random_seed=42,
                             logging_level='Silent',
                             bagging_temperature = 2.0,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20,
                              )


X_train,X_test_eva,y_train,y_test_eva =train_test_split(X_train,y_train,test_size=0.2)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test_eva, y_test_eva, reference=lgb_train)
model = lgb.train(lgb_params, lgb_train, num_boost_round=2500, valid_sets=lgb_eval, early_stopping_rounds=100, verbose_eval=100)
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# cat_model.fit(X_train,y_train,categorical_features_indices)
# cat_y_predict=cat_model.predict(X_test)


# gbr=GradientBoostingRegressor(n_estimators=300, max_depth=7, min_samples_split=210, max_features=17,random_state=10,learning_rate=0.1)
# gbr.fit(X_train,y_train)
# gbr_y_predict=gbr.predict(X_test)

predict = pd.DataFrame(columns=['id','score'])
predict1 = pd.read_csv('test_data.csv')['id']
predict2 = pd.DataFrame(y_pred)

predict=pd.concat([predict1,predict2],axis=1)

predict.columns=['id','score']
predict.to_csv('submission.csv',index=0,float_format='%.2f')

# cv_results = lgb.cv(
#     params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='rmse',
#     early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)


# print('best n_estimators:', len(cv_results['rmse-mean']))
# print('best cv score:', cv_results['rmse-mean'][-1])
#
# #y_pred=rfr_y_predict
# ll=log_loss(y_true,y_pred)
# print(ll)