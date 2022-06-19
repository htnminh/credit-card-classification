
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np



X_train, y_train, X_test, y_test = get_train_test_onehot_df()

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