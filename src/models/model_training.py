# Model training and evaluation module

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from config import RANDOM_FOREST_PARAMS, KFOLD_SPLITS


class ModelTrainer:
    """Class to handle model training and evaluation."""
    
    def __init__(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
        self.models = {
            'LinearRegression': self.lr_model,
            'RandomForest': self.rf_model
        }
        self.predictions = {}
        self.metrics = {}
    
    def train_models(self, X_train, y_train):
        """
        Train ML models.
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        print("\nTraining Linear Regression...")
        self.lr_model.fit(X_train, y_train)
        
        print("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        print("Models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate models on test set
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Linear Regression
        y_pred_lr = self.lr_model.predict(X_test)
        self.predictions['LinearRegression'] = y_pred_lr
        
        lr_mse = mean_squared_error(y_test, y_pred_lr)
        lr_rmse = np.sqrt(lr_mse)
        lr_mae = mean_absolute_error(y_test, y_pred_lr)
        lr_r2 = r2_score(y_test, y_pred_lr)
        
        self.metrics['LinearRegression'] = {
            'MSE': lr_mse,
            'RMSE': lr_rmse,
            'MAE': lr_mae,
            'R2': lr_r2
        }
        
        print("\nLinear Regression Metrics:")
        print(f"  MSE:  {lr_mse:.6f}")
        print(f"  RMSE: {lr_rmse:.6f}")
        print(f"  MAE:  {lr_mae:.6f}")
        print(f"  R²:   {lr_r2:.6f}")
        
        # Random Forest
        y_pred_rf = self.rf_model.predict(X_test)
        self.predictions['RandomForest'] = y_pred_rf
        
        rf_mse = mean_squared_error(y_test, y_pred_rf)
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        rf_r2 = r2_score(y_test, y_pred_rf)
        
        self.metrics['RandomForest'] = {
            'MSE': rf_mse,
            'RMSE': rf_rmse,
            'MAE': rf_mae,
            'R2': rf_r2
        }
        
        print("\nRandom Forest Metrics:")
        print(f"  MSE:  {rf_mse:.6f}")
        print(f"  RMSE: {rf_rmse:.6f}")
        print(f"  MAE:  {rf_mae:.6f}")
        print(f"  R²:   {rf_r2:.6f}")
    
    def cross_validation(self, X_train, y_train):
        """
        Perform k-fold cross validation
        """
        print("\n" + "="*50)
        print("CROSS-VALIDATION (K-Fold, k=5)")
        print("="*50)
        
        kfold = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=100)
        
        for model_name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kfold)
            print(f"\n{model_name} R² scores: {scores}")
            print(f"  Mean: {scores.mean():.6f}")
            print(f"  Std:  {scores.std():.6f}")
    
    def print_summary(self):
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        print("\nR² Scores Comparison:")
        for model_name, metrics in self.metrics.items():
            print(f"  {model_name}: {metrics['R2']:.6f}")
