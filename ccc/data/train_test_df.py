from sklearn.model_selection import train_test_split

from data_df import get_data_df


def get_train_test_df():
    '''
    return (X_train, X_test, y_train, y_test) which are pandas dataframes,
    stratified by y (i.e. the proportion between Target of 0 and 1 are equal
    in train and test)
    '''
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    df = get_data_df()
    y = df['Target'].to_frame()
    X = df.drop(['ID', 'Target'], inplace=False, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    print(get_train_test_df())