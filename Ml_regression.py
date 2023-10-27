import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import probplot
import  datetime as dt
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/anton/Downloads/ML_Project/train.csv", encoding='unicode_escape')
df2 = pd.read_csv("C:/Users/anton/Downloads/ML_Project/test.csv", encoding='unicode_escape')

newdf = df.dropna(axis=0, how='any')
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

X = train.drop([ 'Rented Bike Count', 'Date'], axis=1)
# X['Date'] = pd.to_datetime(X['Date'], dayfirst=True)
# X['Date'] = X['Date'].map(dt.datetime.toordinal)
y = train['Rented Bike Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=((2760)/(2760+6000)), random_state=20)
norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)
ridge = Ridge(alpha=0.6)
ridge.fit(X_train_norm, y_train)
y_pred_ridge = ridge.predict(X_test_norm)
print(mean_squared_error(y_test, y_pred_ridge, squared=False))
# g = pd.DataFrame(train['Visibility (10m)']).copy()
#
# for i in g.columns:
#     probplot(x=g[i], dist='norm', plot=plt)
#     plt.title(i)
#     # plt.show()
# shuffled_train = train.sample(frac=1, random_state=42).reset_index(drop=True)
# shuffled_test = test.sample(frac=1, random_state=42).reset_index(drop=True)
# X_train = shuffled_train.drop(['Date', 'Rented Bike Count'], axis=1)
# y_train = shuffled_train['Rented Bike Count']
# X_test = shuffled_test.drop(['ID', 'Date'], axis=1)
# y_test = pd.DataFrame(test['ID']).copy()
# y_test['Rented Bike Count'] = 0
# norm = MinMaxScaler().fit(X_train)
# X_train_norm = norm.transform(X_train)
# X_test_norm = norm.transform(X_test)
# ridge = Ridge(alpha=0.01)
# ridge.fit(X_train_norm, y_train)
# y_pred = ridge.predict(X_test_norm)
lasso = Lasso(alpha=0.005)
lasso.fit(X_train_norm, y_train)
y_pred_lasso = lasso.predict(X_test_norm)
print(mean_squared_error(y_test, y_pred_lasso, squared=False))
EN = ElasticNet(alpha=0.0011)
EN.fit(X_train_norm, y_train)
y_pred_EN = EN.predict(X_test_norm)
print(mean_squared_error(y_test, y_pred_EN, squared=False))
LR = LinearRegression()
LR.fit(X_train_norm, y_train)
y_pred_LR = LR.predict(X_test_norm)
print(mean_squared_error(y_test, y_pred_LR, squared=False))