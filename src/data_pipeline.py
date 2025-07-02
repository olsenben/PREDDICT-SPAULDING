from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

def drop_mostly_empty_columns(df, threshold=0.9):

    missing_ratio = df.isna().mean()
    dropped_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    df_filtered = df.drop(columns=dropped_cols)

    return df_filtered, dropped_cols



def preprocess_df(df, ordinal_col, nominal_col, continuous_col, ordinal_pipeline,categorical_pipeline,numerical_pipeline,threshold=0.8):
    

    #remove  inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_filtered, drop_cols = drop_mostly_empty_columns(df, threshold)

    ordinal_col = [col for col in ordinal_col if col not in drop_cols]

    nominal_col = [col for col in nominal_col if col not in drop_cols]

    continuous_col = [col for col in continuous_col if col not in drop_cols]

    #combine
    preprocessor = ColumnTransformer([
        ('ord', ordinal_pipeline, ordinal_col),
        ('cat', categorical_pipeline, nominal_col),
        ('num', numerical_pipeline, continuous_col)
    ])

    encoded_df = preprocessor.fit_transform(df_filtered)

    column_names = preprocessor.get_feature_names_out()
    encoded_df = pd.DataFrame(encoded_df, columns=column_names)

    cat_cols = [name for name in column_names if name.startswith('cat__')]
    ord_cols = [name for name in column_names if name.startswith('ord__')]
    num_cols = [name for name in column_names if name.startswith('num__')]


    encoded_df[num_cols] = encoded_df[num_cols].astype(float)
    encoded_df[cat_cols] = encoded_df[cat_cols].astype(str).astype('category') 

    for col in ord_cols:
        encoded_df[col] = pd.Categorical(encoded_df[col], ordered=True)

    #check for NaN values
    assert not encoded_df.isna().any().any(), "Still contains NaNs"
    assert np.isfinite(encoded_df.select_dtypes(include=[np.number])).all().all(), "Still contains infs"
    print(f"Remaining rows: {df.shape[0]}")

    #double check for mixed data columns
    for col in encoded_df.columns:
        types = encoded_df[col].map(type).nunique()
        if types > 1:
            print(f"{col} has mixed types")
            encoded_df[col] = encoded_df[col].astype(str)


    return encoded_df
