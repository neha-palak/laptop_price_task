import pandas as pd
import numpy as np

def preprocess_data(file_path, target_column):
    """
    Preprocess the dataset:
    - Handle missing values
    - Encode categorical variables
    - Standardize numeric features
    - Separate features and target

    Args:
        file_path (str): Path to the CSV file
        target_column (str): Name of the target column for this dataset

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        feature_names (list): Names of features (after encoding)
    """
    
    # Load data
    df = pd.read_csv(file_path)

    # Handle missing values
    for col in df.columns:
        if col == target_column:
            continue  # Don't impute the target variable
        if df[col].dtype == 'object':  
            df[col] = df[col].fillna("Unknown")   # Fill missing categorical with 'Unknown'
        else:
            df[col] = df[col].fillna(df[col].mean())  # Fill missing numeric with mean

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Standardize numeric columns (optional but helps with regression)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_column]  # exclude target
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:  # avoid division by zero
            df[col] = (df[col] - mean) / std

    # Separate features and target
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    feature_names = df.drop(target_column, axis=1).columns.tolist()

    return X, y, feature_names
