
# Step 1: Data Cleaning & Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Load data
df = pd.read_csv("startup_cleaned.csv")  # Make sure this file is in the same directory

# Fill missing subverticals with 'Unknown'
df['subvertical'].fillna('Unknown', inplace=True)

# Extract year and month from date
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# New feature: number of investors
df['num_investors'] = df['investors'].apply(lambda x: len(str(x).split(',')))

# Select features and target
X = df[['vertical', 'subvertical', 'city', 'round', 'year', 'month', 'num_investors']]
y = df['amount']

# Categorical and numerical columns
cat_cols = ['vertical', 'subvertical', 'city', 'round']
num_cols = ['year', 'month', 'num_investors']

# Preprocessing: One-hot encoding for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'  # Keep numeric columns as is
)

# Pipeline: Preprocessing + Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model and feature pipeline
joblib.dump(model, "funding_model.pkl")
