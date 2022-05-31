import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import column_or_1d
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


np.set_printoptions(edgeitems=30, linewidth=100000,
formatter=dict(float=lambda x: "%.3g" % x))
# SVM Classifier
class SVM:
    def __init__(self):
        #Model
        self.clf = 0
        self.c = 0
        self.g = 0
        # Datasets
        self.df = pd.DataFrame
        #Data
        self.X = pd.DataFrame
        #Target
        self.Y = pd.DataFrame
        #Train
        self.X_train =pd.DataFrame
        self.Y_train =pd.DataFrame
        #Test
        self.X_test =pd.DataFrame
        self.Y_test = pd.DataFrame
        self.Y_predict = pd.DataFrame

    def __str__(self):
        return "this is svm "

    def getParameter(self):
        '''return parameter C, and gammar of the model'''
        return (self.c, self.g)

    def getDataset(self,file: str):
        '''Collecting data step
           --------------------
           file: directory of a csv/json file
           --------------------'''
        try:
            for i in range(len(os.path.dirname(__file__))):
                if os.path.dirname(__file__)[-i] == 'c':
                    file = os.path.dirname(__file__)[:-i+1] + "\\data\\"+file
                    break
            print(file)
            l = len(file)
            if file[l-4:l]== ".csv":
                self.df = pd.read_csv(file)
            elif file[l-5: l] == ".json" :
                self.df = pd.read_json(file)
            else:
                raise Exception
        except FileNotFoundError:
            print('Cannot find the file')
        except Exception as e:
            print("The file is not csv or json")

    def dataPrepare(self, test_size =0.2, random_state = 1, dimention_reduct = False, k = 2, feature_select = False, f_type = ('sfm',0.01), oversampling = False, encoding= 22 ) :
        '''preprocess data- oversampling, handling missing data, encoding, splitting,
           feature-selecting(linearSVC), dimention-reducting(pca)
           --------------------------------
           test-size: int - size of testset
           random_state: int - parameter for splitting
           dimention_reduct: bool -dimention_reduct or not
           k: int - number of dimention after dimention_reduct( only use if dimention_reduct = True)
           feature_select: bool -feature_select or not
           f_type: tuple - ('sfm',C) or ('rfe',k) - feature_select type sfm with linearSVC parameter C (C>>, feature_selected >>)
                                                    rfe with k feature chosen ( k can not higher than number of columns)
           oversampling: bool - oversampling or not if the data's imbalanced
           endcoding : int - 11: label encoding for X
                             12: label encoding for Y
                             13: label encoding for X,Y
                             21: onehot encoding for X
                             22: none
                             23: onehot encoding for X label encoding for Y
           -------------------------------'''

        #Seperate
        self.X = self.df.iloc[:,:-1]
        self.Y = self.df.iloc[:,-1]

        #Data Cleaning( missing data)
        #missing data ( impute mean value for numeric and most_frequent for else)
        mf_imputer  = SimpleImputer(missing_values = None, strategy = "most_frequent")
        for i in self.X.columns:
            if pd.api.types.is_numeric_dtype(self.X[i]):
                self.X[i].fillna(self.X[i].mean(), inplace = True)
        self.X = pd.DataFrame(mf_imputer.fit_transform(self.X), columns=self.X.columns).apply(pd.to_numeric, errors='ignore', axis=1)

        #Data Transformation (encoding)
        #Encoding (for different problems use differents type)
        le =LabelEncoder()
        if encoding == 21:
            self.X = pd.get_dummies(self.X) # one hot encoding
        if encoding == 11:
            for col in self.X.columns:
                if not pd.api.types.is_numeric_dtype(self.X[col]):
                    self.X[col] = le.fit_transform(self.X[col])
        if encoding == 12:
            self.Y = pd.DataFrame( le.fit_transform(self.Y))
        if encoding == 13:
            for col in self.X.columns:
                if not pd.api.types.is_numeric_dtype(self.X[col]):
                    self.X[col] = le.fit_transform(self.X[col])
            self.Y = pd.DataFrame(le.fit_transform(self.Y))
        if encoding == 23:
            self.X = pd.get_dummies(self.X) # one hot encoding
            self.Y = pd.DataFrame(le.fit_transform(self.Y))

        #Test_set and Train_set
        self.X_train, self.X_test, self.Y_train, self.Y_test= train_test_split(self.X.to_numpy(), self.Y.to_numpy(), test_size= test_size, random_state=random_state)

        #Data Transformation (standardlising data)
        #Data Standardlizing
        st_x = StandardScaler()
        self.X_train = pd.DataFrame(st_x.fit_transform(self.X_train), columns=self.X.columns)
        self.X_test = pd.DataFrame(st_x.fit_transform(self.X_test), columns=self.X.columns)
        #print(self.X_train, self.Y_train)
        self.Y_train =pd.DataFrame(self.Y_train,dtype= int)
        self.Y_test =pd.DataFrame( self.Y_test, dtype = int)

        #Data Reduction ( dimentional reduction, feature selection)
        if feature_select == True:
            #Feature selection for trainset
            if f_type[0] == 'sfm':
                lsvc = LinearSVC(C=f_type[1], penalty="l1", dual=False).fit(self.X_train, self.Y_train)
                model = SelectFromModel(lsvc, prefit=True)
                model_train =model.fit(self.X_train)
                model_test = model.fit(self.X_test)
            if f_type[0] =='rfe':
                estimator = SVR(kernel="linear")
                model = RFE(estimator, n_features_to_select=f_type[1], step=1)
                model_train =model.fit(self.X_train,self.Y_train)
                model_test = model.fit(self.X_test,self.Y_test)
            self.X_train = pd.DataFrame(model_train.transform(self.X_train), columns= model_train.get_feature_names_out())
            #Feature selection for testset
            self.X_test = pd.DataFrame(model_test.transform(self.X_test), columns= model_test.get_feature_names_out())
        if dimention_reduct ==True:
            #Dimentional reduction for trainset
            pca = PCA(n_components=k)
            principalComponents_train = pca.fit_transform(self.X_train)
            self.X_train = pd.DataFrame(data = principalComponents_train
                          , columns = ['principal component '+ str(i) for i in range(1,k+1)])
            #Dimentional reduction for trainset
            principalComponents_test = pca.fit_transform(self.X_test)
            self.X_test = pd.DataFrame(data = principalComponents_test
                        , columns = ['principal component ' +str(i) for i in range(1,k+1)])

        #Oversampling(SMOTE)
        if oversampling == True:
            smote = SMOTE()
            self.X_train,self.Y_train = smote.fit_resample(self.X_train, self.Y_train)



    def visualization(self, column: int, type: str, column2 = 1 ):
        '''Visualize data as bar chart
        -------------------
        column: int - index of the column
        type: bar - bar chart
              scatter - scatter plot
        column2: int- index of the column2 (only use with scatter plot)
        ------------------- '''
        plt.style.use('seaborn-poster')
        plt.figure(figsize =(8,8))
        if type == 'bar':
            self.X.value_counts(self.X_train.columns[column]).plot(kind = 'bar')
        elif type == 'scatter':
            for i, c, s in (zip(range(len(set(self.Y_train.to_numpy().reshape(-1,)))), ['b', 'g'], ['o', '^'])):
                ix = self.Y_train.to_numpy().reshape(-1,) == i
                plt.scatter(self.X_train.iloc[:, 0][ix], self.X_train.iloc[:, 1][ix], color=c, marker=s, s=60, label=list(set(self.Y_train.to_numpy().reshape(-1,)))[i])
            plt.legend(loc=2, scatterpoints=1)
            plt.xlabel("feature 1-" + self.X_train.columns[column])
            plt.ylabel("feature 2-" + self.X_train.columns[column2])
        plt.show()

    def plot_desicion_boundary(self, title = None):
        '''Visualize classification
        ----------------------
        title: str - name of the scatter chart
        ----------------------'''
        X = self.X.to_numpy()
        y = self.Y.to_numpy().reshape(-1,)
        X0 = X[:, 0]
        X1 = X[:, 1]

        x_min, x_max = X0.min() - 1, X0.max() +1
        y_min, y_max = X1.min() - 1, X1.max() +1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1))

        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize = (10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X0, X1, c=y, alpha=0.8)

        if title is not None:
            plt.title(title)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()



    def train(self, c , g):
        '''train model
        ----------------
        c: regulation of model
        g: gamma
        ----------------'''

        # SVM classifier
        self.c = c
        self.g = g
        self.clf = SVC(kernel = 'rbf', C = c, gamma =g)
        # Train
        self.clf.fit(self.X_train, self.Y_train)


    def predict(self):
        '''make prediction'''
        self.Y_predict = self.clf.predict(self.X_test)

    def accuraccy(self):
        '''return accuraccy score of model'''
        return accuracy_score(self.Y_test, self.Y_predict)

    def f_score(self,k):
        '''return f_1 score of model'''
        return  f1_score(self.Y_test,self.Y_predict, pos_label=k)

    def confusion_matrix(self, visualize = False):
        '''confusion matrix'''
        cfm = confusion_matrix(self.Y_test,self.Y_predict)
        if visualize == True:
             ConfusionMatrixDisplay(cfm).plot()
             plt.show()

        return cfm





