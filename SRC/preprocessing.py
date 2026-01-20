import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(df,missing_value_strategy='median',encode_type = True):
    x = df.drop('quality', axis = 1)
    y = df['quality']

    if missing_value_strategy:
        num_cols = x.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy = missing_value_strategy)
        x[num_cols] = imputer.fit_transform(x[num_cols])


    if encode_type and 'type' in x.columns:
        x['type'] = x['type'].map({'red': 0, 'white': 1})


    assert x.isnull().sum().sum() == 0
    assert x.shape[0] == y.shape[0]

    return x,y