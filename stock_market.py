import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
data = pd.read_csv('infy.csv')
data = data[['Open Price', 'High Price', 'Low Price', 'Close Price']]
print(data)
data['label'] = data['Close Price'].shift(-3)
print(data)
X = np.array(data[['Open Price','High Price','Low Price']])

X_lately = X[-3:]
X = X[:-3]
y = np.array(data['label'])
y = y[:-3]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf=LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('confidence:', confidence)
forecast_set = clf.predict(X_lately)

print(forecast_set)
