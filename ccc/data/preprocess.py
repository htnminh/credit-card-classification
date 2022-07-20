'''
THIS SCRIPT IS USED TO GENERATE *_prep.csv files
FROM X_test.csv, X_train.csv, y_test.csv, AND y_train.csv
THEREFORE ONLY NEED TO RUN ONCE AT THE START OF THE PROJECT.

RANDOM_STATE = 69
'''

import os

import pandas as pd
import numpy as np

import category_encoders as ce

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PowerTransformer

from imblearn.over_sampling import SMOTENC


ABS_PATH = os.path.abspath(os.path.dirname(__file__))
RANDOM_STATE = 69

X_train = pd.read_csv(os.path.join(ABS_PATH, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(ABS_PATH, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(ABS_PATH, 'y_train.csv'))
y_test = pd.read_csv(os.path.join(ABS_PATH, 'y_test.csv'))

y_train = y_train.to_numpy().ravel()
y_test = y_test.to_numpy().ravel()


# Encode education type
education_type_map = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}

X_train['Education_type'] = X_train['Education_type'].map(education_type_map)
X_test['Education_type'] = X_test['Education_type'].map(education_type_map)


# Power transform: Education_type
X_train_filtered = X_train.loc[:, 'Education_type'].to_frame()

transformer = PowerTransformer()
X_train_filtered_transformed = transformer.fit_transform(X_train_filtered)

X_train.loc[:, 'Education_type'] = X_train_filtered_transformed

X_test_filtered_transformed = transformer.transform(
    X_test.loc[:, 'Education_type'].to_frame())

X_test.loc[:, 'Education_type'] = X_test_filtered_transformed


# Power transform: Num_children to Years_employed
X_train_filtered = X_train.loc[:, 'Num_children':'Years_employed']

transformer = PowerTransformer()
X_train_filtered_transformed = transformer.fit_transform(X_train_filtered)

X_train.loc[:, 'Num_children':'Years_employed'] = X_train_filtered_transformed

X_test_filtered_transformed = transformer.transform(
    X_test.loc[:, 'Num_children':'Years_employed'])

X_test.loc[:, 'Num_children':'Years_employed'] = X_test_filtered_transformed


# SMOTENC (still treat Education_type as a categorical_feature)
smote = SMOTENC(categorical_features=list(range(13, 18)),
                # Income_type Education_type Family_status Housing_type Occupation_type
                random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)


# One hot (no more treat Education_type as a categorical_feature)
enc = ce.one_hot.OneHotEncoder()

X_train = enc.fit_transform(X_train, y_train)
X_test = enc.transform(X_test, y_test)


# Write to files
X_train.to_csv(os.path.join(ABS_PATH, 'X_train_prep.csv'), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(ABS_PATH, 'y_train_prep.csv'), index=False)
X_test.to_csv(os.path.join(ABS_PATH, 'X_test_prep.csv'), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(ABS_PATH, 'y_test_prep.csv'), index=False)


# TEST
'''
rf = RandomForestClassifier(n_estimators=50, 
                            max_depth=5,  # seriously overfitting without this param
                            random_state=RANDOM_STATE).fit(X_train, y_train)

print(fbeta_score(y_test, rf.predict(X_test), beta=2))
print(confusion_matrix(y_test, rf.predict(X_test)))
print(fbeta_score(y_train, rf.predict(X_train), beta=2))
'''

"""
0.3342716396903589
[[710 554] 
 [ 98  95]]
0.7717067461161948
"""