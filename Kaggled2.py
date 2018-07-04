import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler, PolynomialFeatures, LabelEncoder
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


#Data Manipulation Here#
temp['temp_LotAreaCut']=pd.qcut(temp.LotArea,10,duplicates='drop')
temp['temp_LotFrontage']=temp.groupby(['temp_LotAreaCut'])['LotFrontage'].transform(lambda x: (x.median()))
temp['LotFrontage']=np.where(temp.LotFrontage.isna(), temp.temp_LotFrontage, temp.LotFrontage)
temp=temp.drop(['temp_LotFrontage','temp_LotAreaCut'],axis=1)


#Normal Sub
temp['YRBltToYrSold']=(temp['YrSold']-temp['YearBuilt']).replace(-1,0)
temp['YRRemodToSold']=(temp['YrSold']-temp['YearRemodAdd']).replace(-1,0)
temp['GarageToSold']=(temp['YrSold']-temp['GarageYrBlt']).replace(-1,0)

temp['TotalSF']=temp['TotalBsmtSF']+temp['1stFlrSF']+temp['2ndFlrSF']

#Log Transform
temp['log_YRBltToYrSold']=(temp['YrSold']-temp['YearBuilt']).apply(np.log).replace(-np.inf,0)
temp['log_GarageToSold']=(temp['YrSold']-temp['GarageYrBlt']).apply(np.log).replace(-np.inf,0)


Str = ["MSSubClass","BsmtFullBath","BsmtHalfBath","HalfBath","BedroomAbvGr","MoSold", 'OverallCond','KitchenAbvGr']
ordered_lbl_columns = [
        'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold']
numeric_feats = ['LotArea'
    ,'MiscVal','PoolArea','LowQualFinSF','3SsnPorch','LandSlope'
    #             ,'BsmtFinSF2','EnclosedPorch','ScreenPorch','MasVnrArea','BsmtFinSF1','TotalBsmtSF'
                 ,'1stFlrSF','GrLivArea','WoodDeckSF','OpenPorchSF','TotalSF']
for i in Str:
    temp[i]=temp[i].fillna('None')
    temp[i]=temp[i].astype(str)
for i in ordered_lbl_columns:
    temp[i].fillna(0,inplace=True)
    lbl = LabelEncoder()
    lbl.fit(list(temp[i].values))
    temp[i] = lbl.transform(list(temp[i].values))

lam = 0.15
for i in numeric_feats:
    abc=str(skew((temp[i].dropna())))
    print(i +":"+abc)
    temp[i] = boxcox1p(temp[i], lam)


temp=temp.fillna(0)
temp=pd.get_dummies(temp)

#Use for testing purposes#
train_x=temp.head(1460)
#Getting Rid of an outlier#
train_x['SalePrice']=all_y
train_x=train_x[(train_x['SalePrice']>=300000) | (train_x['GrLivArea'] < 4000)]
all_y=train_x['SalePrice']
train_x=train_x.drop(['SalePrice'],axis=1)
#Modelling
LGB=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5
                              )
#23722
print('starting overall model')
y_train=(all_y)

scaler=MinMaxScaler()
scaler.fit(train_x)
train_x=scaler.transform(train_x)

LGB.fit(train_x,y_train)
print('Overall RMPSE')
cv=cross_validate(LGB,train_x,y_train,scoring=('neg_mean_squared_error'),return_train_score=False,cv=20)
print(np.sqrt(np.abs(np.mean(cv['test_score']))))
pre_preds=LGB.predict(train_x)

#Use when Submitting Below#
'''
test_x=temp.tail(1459)
test_x=scaler.transform(test_x)
LGB.fit(train_x,y_train)
preds=(LGB.predict(test_x))
id_array = list(range(1461,2920))
submission_frame=pd.DataFrame({'id':id_array,'SalePrice':preds})
submission_frame=submission_frame[['id','SalePrice']]
submission_frame.to_csv('out.csv',index=False)
'''