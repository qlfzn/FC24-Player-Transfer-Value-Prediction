# FC24-Player-Transfer-Value-Prediction

Implemented machine learning algorithms to predict player transfer values in EA FC24, a popular football video game, leveraging data-driven approaches to inform player acquisition decisions and optimise team management strategies. 

Objective: The project aimed to assist gamers in making informed choices about player transfers within the game's virtual marketplace.

# Data Resource
Kaggle: https://www.kaggle.com/datasets/mdkabinhasan/trending-fifa-players-dataset

# Methods
1. Data ingestion
   - Read data from CSV file
   - Load into pandas DataFrame

2. Data wrangling
   - Find information and details of the data
   - Check for uncleaned data - null values, duplicates if necessary
   - Clean data within the columns

3. Exploratory Data Analysis (EDA)
   - Visualise the correlations between features
   - Check patterns and trends in relationships between features

4. Data Preprocessing
   - Normalise and scale the data using Min-Max scaling method
   - Split data to training and test sets

5. Model Training & Evaluation
   - Import ML algorithms for model training - RandomForestRegresser & LinearRegression
   - Fit the data and train the algorithms
   - Compute the training performance using suitable metrics and evaluate
   - Check for possibility of overfitting by computing R-squared score
  
# Findings
R-squared scores: 

LinearRegression :  0.7440433497125863
RandomForestRegressor :  0.9615591064665927

As the result, RandomForestRegressor's predictions are closer to the actual values than LogisticRegression

