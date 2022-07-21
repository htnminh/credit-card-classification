import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# Import the data
df = pd.read_csv('ccc/data/clean_data.csv')

# import sets
DIR_PATH = "ccc/data/"
X_train = pd.read_csv(os.path.join(DIR_PATH, 'X_train_prep.csv')) 
X_test = pd.read_csv(os.path.join(DIR_PATH, 'X_test_prep.csv'))
y_train = pd.read_csv(os.path.join(DIR_PATH, 'y_train_prep.csv'))
y_test = pd.read_csv(os.path.join(DIR_PATH, 'y_test_prep.csv'))
y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.6,
    random_state=RANDOM_STATE,
)

# RICE TUNING
f2_score = lambda y_test, y_pred: fbeta_score(y_test, y_pred, beta=2)
f2_scorer = make_scorer(fbeta_score, beta=2)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    max_leaf_nodes=50,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
y_val_pred = rf.predict(X_val)

# Random Forest F2 Score
print('F2 score: ', f2_score(y_train, rf.predict(X_train)))
print('F2 score: ', f2_score(y_val, y_val_pred))

# Confusion Matrix
print('Confusion Matrix: \n', confusion_matrix(y_val, y_val_pred))

# Classification Report
print('Classification Report: \n', classification_report(y_val, y_val_pred))

# plot_roc_curve(rf, X_train, y_train)
# plot_roc_curve(rf, X_test, y_test)
# plt.show()

# tune the model
from sklearn.model_selection import GridSearchCV

