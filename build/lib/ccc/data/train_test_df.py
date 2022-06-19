from ccc.data.data_df import get_data_df

from sklearn.model_selection import train_test_split
import pandas as pd


def get_train_test_df():
    '''
    return (X_train, X_test, y_train, y_test) which are pandas dataframes,
    stratified by y (i.e. the proportions between Target of 0 and 1 are equal
    in train and test)
    '''
    TEST_SIZE = 0.15
    RANDOM_STATE = 69

    df = get_data_df()
    y = df['Target'].to_frame()
    X = df.drop(['ID', 'Target'], inplace=False, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (X_train, y_train, X_test, y_test)

def get_train_test_onehot_df():
    '''TODO'''
    TEST_SIZE = 0.15
    RANDOM_STATE = 69

    df = get_data_df()
    y = df['Target'].to_frame()
    X = df.drop(['ID', 'Target'], inplace=False, axis=1)

    X = pd.get_dummies(X)
    y = pd.get_dummies(y)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    
    X_train, y_train, X_test, y_test = get_train_test_df()
    print(X_train, y_train, X_test, y_test)