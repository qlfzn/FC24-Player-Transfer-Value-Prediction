"""
Orchestrates the entire ML pipeline for predicting
player transfer values using Linear Regression and Random Forest models.
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_preprocessing import load_data, explore_data, preprocess_data
from preprocessing.feature_engineering import scale_features, prepare_train_test_data
from models.model_training import ModelTrainer
from visualization import perform_eda, plot_model_comparisons

warnings.filterwarnings("ignore")

plt.style.use('default')
sns.set_style("whitegrid")


def main():
    """Main pipeline execution."""
    print("\n" + "="*50)
    print("FIFA PLAYER TRANSFER VALUE PREDICTION")
    print("="*50)
    
    print("\n[1/6] Loading data...")
    df = load_data()
    explore_data(df)
    
    print("\n[2/6] Preprocessing data...")
    df = preprocess_data(df)
    print("Data preprocessing completed!")
    
    print("\n[3/6] Performing exploratory data analysis...")
    perform_eda(df)
    
    print("\n[4/6] Scaling features...")
    scaled_df = scale_features(df)
    print("Features scaled successfully!")
    
    print("\n[5/6] Preparing training and test data...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(scaled_df)
    
    print("\n[6/6] Training and evaluating models...")
    trainer = ModelTrainer()
    trainer.train_models(X_train, y_train)
    trainer.evaluate_models(X_test, y_test)
    trainer.cross_validation(X_train, y_train)
    trainer.print_summary()
    
    print("\nGenerating prediction visualizations...")
    plot_model_comparisons(y_test, trainer.predictions['LinearRegression'], 
                          trainer.predictions['RandomForest'])
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
