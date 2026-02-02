# Data visualization module

import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    # only set correlation for numerical columns
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()


def plot_player_category_vs_stats(df):
    sns.boxplot(data=df, x='Player_category', y='Total_stats')
    plt.xlabel('Player Category (0: Defence, 1: Offence)')
    plt.ylabel('Total Stats')
    plt.title('Total Stats by Player Category')
    plt.show()


def plot_overall_vs_value(df):
    sns.lineplot(data=df, x='Overall', y='Value')
    plt.title('Player Transfer Value by Rating')
    plt.xlabel('Rating')
    plt.ylabel('Transfer Value')
    plt.show()


def plot_potential_vs_value(df):
    sns.lineplot(data=df, x='Potential_overall', y='Value')
    plt.title('Player Transfer Value by Potential')
    plt.xlabel('Potential')
    plt.ylabel('Transfer Value')
    plt.show()


def plot_stats_vs_value(df):
    sns.lineplot(data=df, x='Total_stats', y='Value')
    plt.title('Player Transfer Value by Total Stats')
    plt.xlabel('Stats')
    plt.ylabel('Transfer Value')
    plt.show()


def plot_wage_vs_value(df):
    sns.lineplot(data=df, x='Wage', y='Value')
    plt.title('Player Transfer Value by Wage')
    plt.xlabel('Wage')
    plt.ylabel('Transfer Value')
    plt.show()


def plot_model_comparisons(y_actual, y_pred_lr, y_pred_rf):
    """
    Plot actual vs predicted values for models performance.
    """
    plt.figure(figsize=(12, 6))
    
    # Linear Regression
    plt.subplot(1, 2, 1)
    plt.scatter(y_actual, y_pred_lr, color='blue', alpha=0.5)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 
             color='red', linewidth=2, label='Perfect Prediction')
    plt.title('Linear Regression: Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    
    # Random Forest
    plt.subplot(1, 2, 2)
    plt.scatter(y_actual, y_pred_rf, color='green', alpha=0.5)
    plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 
             color='red', linewidth=2, label='Perfect Prediction')
    plt.title('Random Forest: Actual vs. Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def perform_eda(df):
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    plot_correlation_heatmap(df)
    plot_player_category_vs_stats(df)
    plot_overall_vs_value(df)
    plot_potential_vs_value(df)
    plot_stats_vs_value(df)
    plot_wage_vs_value(df)