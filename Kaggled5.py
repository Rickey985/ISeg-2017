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


def map_values():
    temp["oMSSubClass"] = temp.MSSubClass.map({'180': 1,
                                               '30': 2, '45': 2,
                                               '190': 3, '50': 3, '90': 3,
                                               '85': 4, '40': 4, '160': 4,
                                               '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
                                               '120': 6, '60': 6})

    temp["oMSZoning"] = temp.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})

    temp["oNeighborhood"] = temp.Neighborhood.map({'MeadowV': 1,
                                                   'IDOTRR': 2, 'BrDale': 2,
                                                   'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
                                                   'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
                                                   'NPkVill': 5, 'Mitchel': 5,
                                                   'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
                                                   'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
                                                   'Veenker': 8, 'Somerst': 8, 'Timber': 8,
                                                   'StoneBr': 9,
                                                   'NoRidge': 10, 'NridgHt': 10})

    temp["oCondition1"] = temp.Condition1.map({'Artery': 1,
                                               'Feedr': 2, 'RRAe': 2,
                                               'Norm': 3, 'RRAn': 3,
                                               'PosN': 4, 'RRNe': 4,
                                               'PosA': 5, 'RRNn': 5})

    temp["oBldgType"] = temp.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})

    temp["oHouseStyle"] = temp.HouseStyle.map({'1.5Unf': 1,
                                               '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
                                               '1Story': 3, 'SLvl': 3,
                                               '2Story': 4, '2.5Fin': 4})

    temp["oExterior1st"] = temp.Exterior1st.map({'BrkComm': 1,
                                                 'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
                                                 'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
                                                 'BrkFace': 4, 'Plywood': 4,
                                                 'VinylSd': 5,
                                                 'CemntBd': 6,
                                                 'Stone': 7, 'ImStucc': 7})

    temp["oMasVnrType"] = temp.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})

    temp["oExterQual"] = temp.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    temp["oFoundation"] = temp.Foundation.map({'Slab': 1,
                                               'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
                                               'Wood': 3, 'PConc': 4})

    temp["oBsmtQual"] = temp.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})

    temp["oBsmtExposure"] = temp.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})

    temp["oHeating"] = temp.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})

    temp["oHeatingQC"] = temp.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    temp["oKitchenQual"] = temp.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

    temp["oFunctional"] = temp.Functional.map(
        {'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})

    temp["oFireplaceQu"] = temp.FireplaceQu.map({'None': 1, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    temp["oGarageType"] = temp.GarageType.map({'CarPort': 1, 'None': 1,
                                               'Detchd': 2,
                                               '2Types': 3, 'Basment': 3,
                                               'Attchd': 4, 'BuiltIn': 5})

    temp["oGarageFinish"] = temp.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})

    temp["oPavedDrive"] = temp.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})

    temp["oSaleType"] = temp.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
                                           'CWD': 2, 'Con': 3, 'New': 3})

    temp["oSaleCondition"] = temp.SaleCondition.map(
        {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

    temp["TotalHouse"] = temp["TotalBsmtSF"] + temp["1stFlrSF"] + temp["2ndFlrSF"]
    temp["TotalArea"] = temp["TotalBsmtSF"] + temp["1stFlrSF"] + temp["2ndFlrSF"] + temp["GarageArea"]

    temp["+_TotalHouse_OverallQual"] = temp["TotalHouse"] * temp["OverallQual"]
    temp["+_GrLivArea_OverallQual"] = temp["GrLivArea"] * temp["OverallQual"]
    temp["+_oMSZoning_TotalHouse"] = temp["oMSZoning"] * temp["TotalHouse"]
    temp["+_oMSZoning_OverallQual"] = temp["oMSZoning"] + temp["OverallQual"]
    temp["+_oMSZoning_YearBuilt"] = temp["oMSZoning"] + temp["YearBuilt"]
    temp["+_oNeighborhood_TotalHouse"] = temp["oNeighborhood"] * temp["TotalHouse"]
    temp["+_oNeighborhood_OverallQual"] = temp["oNeighborhood"] + temp["OverallQual"]
    temp["+_oNeighborhood_YearBuilt"] = temp["oNeighborhood"] + temp["YearBuilt"]
    temp["+_BsmtFinSF1_OverallQual"] = temp["BsmtFinSF1"] * temp["OverallQual"]

    temp["-_oFunctional_TotalHouse"] = temp["oFunctional"] * temp["TotalHouse"]
    temp["-_oFunctional_OverallQual"] = temp["oFunctional"] + temp["OverallQual"]
    temp["-_LotArea_OverallQual"] = temp["LotArea"] * temp["OverallQual"]
    temp["-_TotalHouse_LotArea"] = temp["TotalHouse"] + temp["LotArea"]
    temp["-_oCondition1_TotalHouse"] = temp["oCondition1"] * temp["TotalHouse"]
    temp["-_oCondition1_OverallQual"] = temp["oCondition1"] + temp["OverallQual"]

    temp["Bsmt"] = temp["BsmtFinSF1"] + temp["BsmtFinSF2"] + temp["BsmtUnfSF"]
    temp["Rooms"] = temp["FullBath"] + temp["TotRmsAbvGrd"]
    temp["PorchArea"] = temp["OpenPorchSF"] + temp["EnclosedPorch"] + temp["3SsnPorch"] + temp["ScreenPorch"]
    temp["TotalPlace"] = temp["TotalBsmtSF"] + temp["1stFlrSF"] + temp["2ndFlrSF"] + temp["GarageArea"] + temp["OpenPorchSF"] + temp[
        "EnclosedPorch"] + temp["3SsnPorch"] + temp["ScreenPorch"]

    return "Done!"

map_values()

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


all_y=train_x['SalePrice']
train_x=train_x.drop(['SalePrice'],axis=1)
#Modelling
print('starting overall model')
y_train=np.log1p(all_y)
rf=Ridge(alpha=2.5,tol=0.00001)
lass=Lasso(alpha=0.0005,random_state=5,max_iter=9999)
#999999999
Elas=ElasticNet(alpha=0.0008,random_state=5,max_iter=99999)
gb=GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
kn=KNeighborsRegressor(n_jobs=3,n_neighbors=5)
str=StackingRegressor(regressors=[lass,gb,kn,Elas],verbose=1,meta_regressor=rf)
print('Overall RMPSE')
cv=cross_validate(str,train_x,y_train,scoring=('neg_mean_squared_error'),return_train_score=False,cv=10)
print(np.sqrt(np.abs(np.mean(cv['test_score']))))

#Use when Submitting Below#

test_x=temp.tail(1459)
str.fit(train_x,y_train)
preds=np.expm1(str.predict(test_x))
id_array = list(range(1461,2920))
submission_frame=pd.DataFrame({'id':id_array,'SalePrice':preds})
submission_frame=submission_frame[['id','SalePrice']]
submission_frame.to_csv('out.csv',index=False)


gb.fit(train_x,y_train)
new_preds=np.expm1(gb.predict(test_x))

dat_frame=pd.DataFrame({'Sub':preds,'GBM':new_preds})