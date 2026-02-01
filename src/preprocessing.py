import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def split_features_target(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
