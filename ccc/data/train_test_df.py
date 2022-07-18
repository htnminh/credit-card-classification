'''
THIS SCRIPT IS USED TO GENERATE X_test.csv, X_train.csv, y_test.csv, AND y_train.csv
FROM clean_data.csv THEREFORE ONLY NEED TO RUN ONCE AT THE START OF THE PROJECT.

RANDOM_STATE = 69
'''


import os
import pandas as pd

from sklearn.model_selection import train_test_split
import pandas as pd


ABS_PATH = os.path.abspath(os.path.dirname(__file__))



def get_data_df():
    '''return the pandas dataframe of clean_data.csv'''
    df = pd.read_csv(os.path.join(ABS_PATH, 'clean_data.csv'))
    
    return df

def split_train_test_df(df):
    '''
    return (X_train, X_test, y_train, y_test) which are pandas dataframes,
    stratified by y (i.e. the proportions between Target of 0 and 1 are equal
    in train and test)
    '''
    TEST_SIZE = 0.15
    RANDOM_STATE = 69

    y = df['Target'].to_frame()
    X = df.drop(['ID', 'Target'], inplace=False, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    df = get_data_df()
    print(df.head())
    print(df.info())

    X_train, y_train, X_test, y_test = split_train_test_df(df)
    print(X_train, y_train, X_test, y_test)

    X_train.to_csv(os.path.join(ABS_PATH, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(ABS_PATH, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(ABS_PATH, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(ABS_PATH, 'y_test.csv'), index=False)
