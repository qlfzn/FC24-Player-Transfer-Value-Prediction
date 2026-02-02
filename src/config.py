import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'updated_trending_football_players.xlsx')

RANDOM_FOREST_PARAMS = {
    'n_estimators': 300,
    'max_features': 'sqrt',
    'max_depth': 5,
    'random_state': 18
}

TEST_SIZE = 0.3
RANDOM_STATE = 42
KFOLD_SPLITS = 5

NUMERICAL_COLS = ['Age', 'Overall', 'Potential_overall', 'Value', 'Wage', 'Total_stats']
TARGET_COL = 'Value'

DEFENSIVE_POSITIONS = ('GK', 'CB', 'RB', 'RWB', 'LB', 'LWB')
