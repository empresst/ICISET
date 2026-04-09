# src/utils/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_datetime_features(df: pd.DataFrame, datetime_col: str = "date_time"):
    """Exact feature engineering from both notebooks"""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df['year'] = df[datetime_col].apply(lambda x: x.year)
    df['quarter'] = df[datetime_col].apply(lambda x: x.quarter)
    df['month'] = df[datetime_col].apply(lambda x: x.month)
    df['day'] = df[datetime_col].apply(lambda x: x.day)
    df['hour'] = df[datetime_col].apply(lambda x: x.hour)
    df['weekday'] = df[datetime_col].apply(lambda x: x.weekday() < 5).astype(int)
    return df

def create_dataset(dataset, dates, look_back=30):
    """Exact function from WithoutPreprprocessLBG.ipynb"""
    X, Y, d = [], [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
        d.append(dates[i + look_back, 0])
    return np.array(X), np.array(Y), np.array(d)

def preprocess_data(df: pd.DataFrame, target_col: str = "Global_active_power", look_back: int = 30):
    """Full preprocessing pipeline from both notebooks - NO CODE REMOVED"""
    # Data cleaning
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])

    # DateTime conversion and feature engineering
    if 'date_time' in df.columns:
        df = add_datetime_features(df)
    elif 'V1' in df.columns:  # For B.txt file
        df['V1'] = pd.to_datetime(df['V1'])
        df['date_time'] = df['V1']
        df = add_datetime_features(df, 'date_time')
        target_col = "V6"

    # Sorting and reset index
    df.sort_values('date_time', inplace=True, ascending=True)
    df = df.reset_index(drop=True)

    # Scaling
    dataset = df[target_col].values.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Train-test split
    train_size = int(len(dataset) * 0.80)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Dates for error table
    col_dates = df['date_time'].values
    col_dates = np.reshape(col_dates, (-1, 1))
    date_train, date_test = col_dates[0:train_size, :], col_dates[train_size:len(dataset), :]

    # Create sequences
    X_train, Y_train, d_train = create_dataset(train, date_train, look_back)
    X_test, Y_test, d_test = create_dataset(test, date_test, look_back)

    # Reshape for models (exact shape used in your notebooks)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return X_train, Y_train, X_test, Y_test, d_test, scaler, df