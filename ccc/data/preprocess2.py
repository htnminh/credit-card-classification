
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import column_or_1d
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, f1_score, classification_report, fbeta_score, make_scorer,plot_roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE,RFECV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

"""# Colecting Data

"""

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(ABS_PATH, 'clean_data.csv'))

"""# Choose importance feature

-Eliminate the unecessary categories in Num_children and Num_family (very little data when num_children >2, and num_family>4) choose 8 top occupation_type(among 19 category value)
"""

df=df.apply(pd.to_numeric, errors='ignore', axis=1)
top_8 = [x for x in df.Occupation_type.value_counts().sort_values(ascending=False).head(8).index]
for i in range(len(df)):
  if df['Num_children'].values[i]>2:
    df['Num_children'].values[i] = 2
  if df['Num_family'].values[i]>4:
    df['Num_family'].values[i] = 4
  if df.Occupation_type.values[i] not in top_8:
    df.Occupation_type.values[i] ='Other'

categorical_df=pd.DataFrame(df,columns = df.columns ).select_dtypes('object')
categorical_df.nunique()

"""# Encoding

"""

le = LabelEncoder()
df['Education_type'] = le.fit_transform(df['Education_type'])
df = pd.get_dummies(df)
df = df.drop(columns =['ID','Income_type_Student','Housing_type_Co-op apartment','Housing_type_Rented apartment','Housing_type_Office apartment','Occupation_type_Other'])

"""#Standardize data"""

st_x = StandardScaler()

df_num = df[['Account_length','Total_income','Age','Years_employed','Num_family','Num_children','Education_type']]
df_num_scaled = st_x.fit_transform(df_num)
df[df_num.columns] = pd.DataFrame(df_num_scaled, index=df_num.index,columns=df_num.columns)

"""# Seperate target and feature

"""

X = df.iloc[:,:-1]
Y = df.iloc[:,-1]

"""# Seperate train set and test set"""

X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size= 0.2, random_state=1)

"""# Seperate validation test and train for tuning"""

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state=1)

"""# Oversampling train set"""

smote = SMOTE(random_state =1)
X_train,y_train = smote.fit_resample(X_train, y_train)

X_train.to_csv("X_train_prep2.csv")
X_test.to_csv("X_test_prep2.csv")
X_val.to_csv("X_validation_prep.csv")

y_train.to_csv("y_train_prep2.csv")
y_test.to_csv("y_test_prep2.csv")
y_val.to_csv("y_validation_prep.csv")

