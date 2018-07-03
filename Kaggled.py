import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler, PolynomialFeatures
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

all_data=pd.read_csv('train_1.csv')
submit_data=pd.read_csv('test_1.csv')
all_data=all_data.drop(['Id'],axis=1)
submit_data=submit_data.drop(['Id'],axis=1)

all_y=all_data[['SalePrice']]
all_data=all_data.drop(['SalePrice'],axis=1)



#Temp Combined Data:#
temp=all_data.append(submit_data)

temp=pd.get_dummies(temp)


#Use for testing purposes#


train_x=temp.head(1460)
x_train,x_test,y_train,y_test=train_test_split(train_x,all_y,random_state=0)

scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

LGB=LinearSVC(random_state=0)
#27695.546481403795

LGB.fit(x_train,y_train)
preds=LGB.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,preds)))


#Use when Submitting Below#

'''

train_x=temp.head(1460)
test_x=temp.tail(1459)


scaler=MinMaxScaler()
scaler.fit(train_x)
train_x=scaler.transform(train_x)
test_x=scaler.transform(test_x)


LGB=lgb.LGBMRegressor(silent=False,random_state=0,boosting_type='gbdt',num_leaves=1600,max_depth=-1
                       ,n_estimators=3500)

LGB.fit(train_x,all_y)
preds=LGB.predict(test_x)
id_array = list(range(1461,2920))
submission_frame=pd.DataFrame({'id':id_array,'SalePrice':preds})
submission_frame=submission_frame[['id','SalePrice']]
submission_frame.to_csv('out.csv',index=False)
'''