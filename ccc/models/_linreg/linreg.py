
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

X_train = pd.read_csv(os.path.join(ABS_PATH, 'X_train.csv')).to_numpy()
y_train = pd.read_csv(os.path.join(ABS_PATH, 'y_train.csv')).to_numpy()
X_test = pd.read_csv(os.path.join(ABS_PATH, 'X_test.csv')).to_numpy()
y_test = pd.read_csv(os.path.join(ABS_PATH, 'y_test.csv')).to_numpy()

print(X_test, y_test)

'''
linreg = LinearRegression().fit(X_train, y_train)
# print(linreg.coef_)

y_pred = linreg.predict(X_test)
n, _ = y_pred.shape
y_pred = y_pred.reshape((n,))
print(y_pred)

y_pred_0_filter = y_pred < 0.5
y_pred[y_pred_0_filter] = 0
y_pred[np.logical_not(y_pred_0_filter)] = 1
print(y_pred)
print(pd.Series.value_counts(y_pred))

print(f'acc = {accuracy_score(y_test, y_pred)}')
'''