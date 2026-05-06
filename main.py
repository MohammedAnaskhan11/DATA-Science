import os
import pandas as pd
from src.preprocessing import load_data, clean_data, scale_features
from src.analysis import perform_eda, get_top_correlations
from src.model import prepare_data, train_model, evaluate_model, save_model

def run_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, 'dataset', 'raw_data', 'sports_data.csv')
    processed_data_path = os.path.join(base_dir, 'dataset', 'processed_data', 'cleaned_sports_data.csv')
    graphs_dir = os.path.join(base_dir, 'outputs', 'graphs')
    results_dir = os.path.join(base_dir, 'outputs', 'results')
    
    # 1. Load Data
    df = load_data(raw_data_path)
    
    # 2. Preprocess Data
    df_clean = clean_data(df)
    
    # We will scale all features except the target for modeling later
    features_to_scale = [col for col in df_clean.columns if col != 'Market_Value']
    df_processed, scaler = scale_features(df_clean, features_to_scale)
    
    # Save processed data
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # 3. Exploratory Data Analysis
    perform_eda(df_clean, output_dir=graphs_dir)
    
    top_corr = get_top_correlations(df_clean, 'Market_Value')
    print("\nTop Correlations with Market Value:")
    print(top_corr.head(3))
    
    # 4. Model Training and Evaluation
    X, y = prepare_data(df_processed, 'Market_Value')
    model, X_test, y_test = train_model(X, y)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. Save Results
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'model_metrics.txt'), 'w') as f:
        f.write(f"Model: Random Forest Regressor\n")
        f.write(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}\n")
        f.write(f"R-squared (R2): {metrics['R2']:.4f}\n")
    
    save_model(model, results_dir)
    print("\nPipeline execution complete!")

if __name__ == '__main__':
    run_pipeline()
