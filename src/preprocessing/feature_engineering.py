# Feature scaling and data preparation module

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import NUMERICAL_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE


def scale_features(df):
    """
    Scale numerical features using MinMax scaling.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[NUMERICAL_COLS])
    scaled_df = pd.DataFrame(scaled_data, columns=NUMERICAL_COLS)
    
    return scaled_df


def prepare_train_test_data(df):
    """
    Prepare training and testing data.
    """
    X = df.drop([TARGET_COL], axis=1)
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
