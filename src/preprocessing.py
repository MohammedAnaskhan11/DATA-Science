import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads dataset from the given filepath."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the dataframe by handling missing values.
    Fills numerical NaNs with the median and categorical NaNs with mode.
    One-hot encodes categorical variables.
    """
    print("Cleaning data and handling missing values...")
    df_cleaned = df.copy()
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['float64', 'int64']:
            if df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
        else:
            if df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()[0]
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                
    # One-hot encode categorical variables
    categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df_cleaned = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)
            
    return df_cleaned

def scale_features(df, features_to_scale):
    """
    Scales selected features using StandardScaler.
    Returns the dataframe with scaled features and the fitted scaler.
    """
    print("Scaling features...")
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])
    return df_scaled, scaler
