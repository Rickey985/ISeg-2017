import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report
from sklearn.ensemble import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler, PolynomialFeatures, LabelEncoder, MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import  GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn import linear_model
from matplotlib import pyplot as plt
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import  KernelRidge
from sklearn.linear_model import *
import lightgbm as lgb
from mlxtend.regressor import StackingRegressor

all_data=pd.read_csv('train_1.csv')
submit_data=pd.read_csv('test_1.csv')
all_data=all_data.drop(['Id'],axis=1)
submit_data=submit_data.drop(['Id'],axis=1)

all_y=all_data['SalePrice']
all_data=all_data.drop(['SalePrice'],axis=1)

#getting the other df#
all_data=all_data.append(submit_data)
#Temp Combined Data:#
temp=all_data
#Very Important and Useful#


#Data Manipulation (Cleaning) via Grouping Here#
temp['temp_LotAreaCut']=pd.qcut(temp.LotArea,10,duplicates='drop')
temp['LotFrontage']=temp.groupby(['temp_LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
temp['GarageYrBlt']=temp.groupby(['temp_LotAreaCut'])['GarageYrBlt'].transform(lambda x: x.fillna(x.median()))
temp=temp.drop(['temp_LotAreaCut'],axis=1)




#Additional Columns#
temp['YRBltToYrSold']=(temp['YrSold']-temp['YearBuilt']).replace(-1,0)
temp['YRRemodToSold']=(temp['YrSold']-temp['YearRemodAdd']).replace(-1,0)
temp['GarageToSold']=(temp['YrSold']-temp['GarageYrBlt']).replace(-1,0)

temp['TotalSF']=temp['TotalBsmtSF']+temp['1stFlrSF']+temp['2ndFlrSF']


#Log Transform
temp['log_YRBltToYrSold']=(temp['YrSold']-temp['YearBuilt']).apply(np.log1p).replace(-np.inf,0)
temp['log_GarageToSold']=(temp['YrSold']-temp['GarageYrBlt']).apply(np.log1p).replace(-np.inf,0)
temp['log_GrLivArea']=temp['GrLivArea'].apply(np.log1p).replace(-np.inf,0)
temp['log_BsmtUnfSF']=temp['BsmtUnfSF'].apply(np.log1p).replace(-np.inf,0)
temp['log_1stFlrSF']=temp['1stFlrSF'].apply(np.log1p).replace(-np.inf,0)

#Strengthened Tranform
temp['LotArea_TotalBsmtSF']=temp['LotArea']*temp['TotalBsmtSF']
temp['LotArea_TotalSF']=temp['LotArea']*temp['TotalSF']
temp['TotalBsmtSF_TotalSF']=temp['TotalBsmtSF']*temp['TotalSF']
temp['TotalBsmtSF_GrLivArea']=temp['TotalBsmtSF']*temp['GrLivArea']
temp['TotalSF_GrLivArea']=temp['TotalSF']*temp['GrLivArea']
temp['TotalBsmtSF_GarageArea']=temp['TotalBsmtSF']*temp['GarageArea']


#Any Last Second Transformations
#Filling the Below NAs with None
Str = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish",
       "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]

for i in Str:
    temp[i]=temp[i].fillna('None')
    temp[i]=temp[i].astype(str)

#Bad Entries Columns#
print('Bad Columns:')
for i in temp.columns.values:
    missing_values=temp[i].isnull().sum()
    if missing_values > 0:
        print(i+':'+str(missing_values))
        #temp=temp.drop(i,axis=1)

#Skewness funkibusiness
numeric_feats = temp.dtypes[temp.dtypes != "object"].index
skewed_feats = temp[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
temp[skewed_feats] = np.log1p(temp[skewed_feats])

temp=temp.fillna(0)
temp=pd.get_dummies(temp)

#Use for testing purposes#
train_x=temp.head(1460)
#Getting Rid of an outlier#
train_x['SalePrice']=all_y
abc=[4,11,13,20,46,66,70,167,178,185,199, 224,261, 309,313,318, 349,412,423,440,454,477,478, 523,540, 581,588,595,654,
     688, 691, 774, 798, 875, 898,926,970,987,1027,1109, 1169,1182,1239, 1256,1298,1324,1353,1359,1405,1442,1447]
train_x.drop(train_x.index[abc],inplace=True)
train_x=train_x[(train_x['SalePrice']>=300000) | (train_x['GrLivArea'] < 4000)]


#train_x=train_x.query('LotArea_TotalSF < 898002140.0')
all_y=train_x['SalePrice']
train_x=train_x.drop(['SalePrice'],axis=1)
#Modelling
LGB=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5
                              )

print('starting overall model')
y_train=np.log1p(all_y)

scaler=StandardScaler()
scaler.fit(train_x)
#train_x=scaler.transform(train_x)

#LGB.fit(train_x,y_train)
rid=KNeighborsRegressor(n_jobs=3, n_neighbors=4)
rf=LinearRegression()
str=StackingRegressor(regressors=[LGB,rid],verbose=1,meta_regressor=rf)
print('Overall RMPSE')
cv=cross_validate(str,train_x,y_train,scoring=('neg_mean_squared_error'),return_train_score=False,cv=10)
print(np.sqrt(np.abs(np.mean(cv['test_score']))))

#Grabbing Feature Importance#
#rint('grabbing feature importance')
#GB.fit(train_x,y_train)
#eature_df=pd.DataFrame({'Cols':train_x.columns,'Vals':LGB.feature_importances_})
#eature_df=feature_df.sort_values(['Vals'],ascending=[0])



#Use when Submitting Below#
'''
test_x=temp.tail(1459)
test_x=scaler.transform(test_x)
str.fit(train_x,y_train)
preds=np.expm1(str.predict(test_x))
id_array = list(range(1461,2920))
submission_frame=pd.DataFrame({'id':id_array,'SalePrice':preds})
submission_frame=submission_frame[['id','SalePrice']]
submission_frame.to_csv('out.csv',index=False)
'''