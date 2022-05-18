import os
import pandas as pd

def get_data_pandas_df():
    '''return the pandas dataframe of clean_data.csv'''
    abs_path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(abs_path, 'clean_data.csv'))
    return df

if __name__ == '__main__':
    df = get_data_pandas_df()
    print(df.head())
    print(df.info())