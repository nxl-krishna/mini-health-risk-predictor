# src/data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prep_data(filepath):
    # 1. Load Data
    df = pd.read_csv(filepath)
    
    # 2. Handle Hidden Missing Values
    # Columns where '0' indicates missing data, not a valid measurement
    zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    # Replace 0 with NaN so pandas treats them as missing
    df[zero_fields] = df[zero_fields].replace(0, np.nan)
    
    # Impute missing values with Median (more robust to outliers than Mean)
    for col in zero_fields:
        df[col] = df[col].fillna(df[col].median())
        
    # 3. Split Data
    # Stratify is crucial here because the dataset is imbalanced (approx 2:1 ratio of Non-Diabetic to Diabetic)
    # Stratify ensures the train/test split maintains this same ratio.
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Normalize Features
    # Standardization: z = (x - u) / s
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability (optional but good for debugging)
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"Training Data Shape: {X_train_df.shape}")
    print(f"Testing Data Shape: {X_test_df.shape}")
    
    return X_train_df, X_test_df, y_train, y_test, scaler

if __name__ == "__main__":
    # Ensure you have the dataset saved as diabetes.csv
    try:
        load_and_prep_data('../data/diabetes.csv')
    except FileNotFoundError:
        print("Error: diabetes.csv not found. Please check the path.")