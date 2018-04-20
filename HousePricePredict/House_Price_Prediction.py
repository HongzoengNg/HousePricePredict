# coding: utf-8
import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def cv_rmsd(y,y_pred):
    y=y.values
    return np.sqrt(((y_pred-y)**2).sum()/len(y))/y.mean()

def output(filename):
    fw=open(filename,'w')
    fw.write('Id,SalePrice\n\n')
    for i in range(len(test_ID)):
        fw.write(str(test_ID.values[i])+','+str(Y_test_pred[i])+'\n')
    fw.close()

if __name__ == '__main__':
    train=pd.read_csv('train.csv')
    test=pd.read_csv('test.csv')

    #missing data in train dataset
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    # MiscFeature: Miscellaneous feature not covered in other categories

    #dealing with missing train data
    train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
    train = train.drop(train.loc[train['Electrical'].isnull()].index)

    #filling missing test data with median
    test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)
    test=test.fillna(test.median())


    for i in range(train.shape[1]):
        if train.iloc[:,i].dtypes == object:
            lb=LabelEncoder()
            lb.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
            train.iloc[:,i]=lb.transform(list(train.iloc[:,i].values))
            test.iloc[:,i]=lb.transform(list(test.iloc[:,i].values))

    # keep ID for submission
    train_ID = train['Id']
    test_ID = test['Id']
    train=train.drop('Id',axis=1)
    test=test.drop('Id',axis=1)

    Y_train = train['SalePrice']
    X_train = train.drop(['SalePrice'], axis=1)
    regre=RandomForestRegressor(n_estimators=1000,criterion='mse',max_features=40,n_jobs=-1,oob_score=True,random_state=2)
    regre.fit(X_train.values,Y_train.values)

    Y_train_pred=regre.predict(X_train.values)
    print(cv_rmsd(Y_train,Y_train_pred))

    regre0=RandomForestRegressor(n_estimators=100,criterion='mae',n_jobs=-1,oob_score=True)
    regre0.fit(X_train['OverallQual'].values.reshape(-1,1),Y_train.values)
    regre1=RandomForestRegressor(n_estimators=100,criterion='mae',n_jobs=-1,oob_score=True)
    regre1.fit(X_train['YearBuilt'].values.reshape(-1,1),Y_train.values)

    y_pred_byOverallQual=regre0.predict(X_train['OverallQual'].values.reshape(-1,1))
    print('The CV_RMSD of regression with OverallQual:%s'% cv_rmsd(Y_train,y_pred_byOverallQual))
    y_pred_byYearBuilt=regre1.predict(X_train['YearBuilt'].values.reshape(-1,1))
    print('The CV_RMSD of regression with YearBuilt:%s'% cv_rmsd(Y_train,y_pred_byYearBuilt))

    # To make it visiable:

    fig=plt.figure(figsize=(14,10))
    sns.regplot(x=X_train['OverallQual'], y=Y_train,marker='.',color='y')
    x=np.random.randint(low=0,high=11,size=20).reshape(-1,1)
    y=regre0.predict(x)
    sns.regplot(x=x,y=y,fit_reg=False,color='r')

    fig=plt.figure(figsize=(14,10))
    sns.regplot(x=X_train['YearBuilt'], y=Y_train,color='grey',marker='.')
    x=np.random.randint(low=1870,high=2010,size=1000).reshape(-1,1)
    y=regre1.predict(x)
    sns.regplot(x=x,y=y,fit_reg=False,color='r',marker='.')

    Y_test_pred=regre.predict(test.values)
    output('submission.csv')

