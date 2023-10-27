import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
import  datetime as dt

df = pd.read_csv("C:/Users/anton/Downloads/ML_Project/train.csv", encoding='unicode_escape')
df2 = pd.read_csv("C:/Users/anton/Downloads/ML_Project/test.csv", encoding='unicode_escape')

newdf = -df.dropna(axis=0, how='any')
newdf2 = df2.dropna(axis=0, how='any')

newdf.drop_duplicates(inplace=True)
newdf2.drop_duplicates(inplace=True)

Holiday_train = pd.get_dummies(newdf['Holiday'])
Holiday_test = pd.get_dummies(newdf2['Holiday'])

FD_train = pd.get_dummies(newdf['Functioning Day'])
FD_test = pd.get_dummies(newdf2['Functioning Day'])

newdf.drop(['Holiday', 'Functioning Day'], axis=1, inplace=True)
newdf2.drop(['Holiday', 'Functioning Day'], axis=1, inplace=True)

newdf['Spring'] = 0
newdf['Summer'] = 0
newdf['Autumn'] = 0
newdf['Winter'] = 0

newdf2['Spring'] = 0
newdf2['Summer'] = 0
newdf2['Autumn'] = 0
newdf2['Winter'] = 0

for i in range(newdf.shape[0]):
    if newdf.at[i, 'Seasons'] == 'Spring':
        newdf.at[i, 'Spring'] = 1
    elif newdf.at[i, 'Seasons'] == 'Summer':
        newdf.at[i, 'Summer'] = 1
    elif newdf.at[i, 'Seasons'] == 'Autumn':
        newdf.at[i, 'Autumn'] = 1
    elif newdf.at[i, 'Seasons'] == 'Winter':
        newdf.at[i, 'Winter'] = 1

for i in range(newdf2.shape[0]):
    if newdf2.at[i, 'Seasons'] == 'Spring':
        newdf2.at[i, 'Spring'] = 1
    elif newdf2.at[i, 'Seasons'] == 'Summer':
        newdf2.at[i, 'Summer'] = 1
    elif newdf2.at[i, 'Seasons'] == 'Autumn':
        newdf2.at[i, 'Autumn'] = 1
    elif newdf2.at[i, 'Seasons'] == 'Winter':
        newdf2.at[i, 'Winter'] = 1

newdf.drop(['Seasons'], axis=1, inplace=True)
newdf2.drop(['Seasons'], axis=1, inplace=True)

train = pd.concat([newdf, Holiday_train, FD_train], axis=1)
test = pd.concat([newdf2, Holiday_test, FD_test], axis=1)

train.reset_index(inplace=True)
test.reset_index(inplace=True)

train.drop(['index'], axis=1, inplace=True)
test.drop(['index'], axis=1, inplace=True)

shuffled_train = train.sample(frac=1, random_state=20).reset_index(drop=True)
shuffled_test = test.sample(frac=1, random_state=20).reset_index(drop=True)

# shuffled_train['Date'] = pd.to_datetime(shuffled_train['Date'], dayfirst=True)
# shuffled_train['Date'] = shuffled_train['Date'].map(dt.datetime.toordinal)
#
# shuffled_test['Date'] = pd.to_datetime(shuffled_test['Date'], dayfirst=True)
# shuffled_test['Date'] = shuffled_test['Date'].map(dt.datetime.toordinal)

X_train = shuffled_train.drop(['Rented Bike Count', 'Date'], axis=1)
y_train = shuffled_train['Rented Bike Count']
X_test = shuffled_test.drop(['ID', 'Date'], axis=1)
y_test_ridge = pd.DataFrame(test['ID']).copy()
y_test_lasso = pd.DataFrame(test['ID']).copy()
y_test_EN = pd.DataFrame(test['ID']).copy()
y_test_LR = pd.DataFrame(test['ID']).copy()
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
ridge = Ridge(alpha=0.6)
ridge.fit(X_train_norm, y_train)
y_pred_ridge = ridge.predict(X_test_norm)
y_test_ridge['Rented Bike Count'] = y_pred_ridge
# y_test_ridge.to_csv('ridge_result.csv', index=False)
lasso = Lasso(alpha=0.005)
lasso.fit(X_train_norm, y_train)
y_pred_lasso = lasso.predict(X_test_norm)
y_test_lasso['Rented Bike Count'] = y_pred_lasso
# y_test_lasso.to_csv('lasso_result.csv', index=False)
EN = ElasticNet(alpha=0.0011)
EN.fit(X_train_norm, y_train)
y_pred_EN = EN.predict(X_test_norm)
y_test_EN['Rented Bike Count'] = y_pred_EN
# y_test_EN.to_csv('EN_result.csv', index=False)
LR = LinearRegression()
LR.fit(X_train_norm, y_train)
y_pred_LR = LR.predict(X_test_norm)
y_test_LR['Rented Bike Count'] = y_pred_LR
# y_test_LR.to_csv('LR_result.csv', index=False)
# print(y_test_ridge)
# print(y_test_lasso)
# print(y_test_EN)
# print(y_test_LR)