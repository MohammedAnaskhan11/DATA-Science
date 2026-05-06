from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def prepare_data(df, target_col):
    """Splits data into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def train_model(X, y):
    """
    Trains a Random Forest Regressor model.
    Splits data into train and test sets internally.
    Returns the trained model, and the test split (X_test, y_test) for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using MSE and R-squared metrics.
    Returns a dictionary of metrics.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Evaluation - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    return {'MSE': mse, 'R2': r2}

def save_model(model, output_dir):
    """Saves the trained model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
