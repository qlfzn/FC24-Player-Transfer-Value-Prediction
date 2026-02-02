import pandas as pd
from config import DATA_PATH, DEFENSIVE_POSITIONS


def load_data():
    df = pd.read_excel(DATA_PATH)
    return df


def explore_data(df):
    print(f"Dataset shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nData summary:\n{df.describe().transpose()}")
    return df


def clean_data(df):
    # convert type str to int
    # df['Contract_end'] = pd.to_numeric(df['Contract_end'])
    df['Contract_end'].fillna('0', inplace=True)

    df = df.drop('Images', axis=1)
    
    return df


def categorize_player(position):
    """
    Categorize player as defensive (0) or offensive (1) based on position.
    """
    if position in DEFENSIVE_POSITIONS:
        return 0
    else:
        return 1


def add_player_category(df):
    df['Player_category'] = df['Positions'].apply(categorize_player)
    return df


def preprocess_data(df):
    df = clean_data(df)
    df = add_player_category(df)
    return df
