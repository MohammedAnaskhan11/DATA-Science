import pandas as pd
import numpy as np
import os
from datetime import datetime

def prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    transfermarkt_dir = os.path.join(base_dir, 'unzipped_data', 'football-datasets-main', 'datalake', 'transfermarkt')
    
    profiles_path = os.path.join(transfermarkt_dir, 'player_profiles', 'player_profiles.csv')
    market_value_path = os.path.join(transfermarkt_dir, 'player_latest_market_value', 'player_latest_market_value.csv')
    national_perf_path = os.path.join(transfermarkt_dir, 'player_national_performances', 'player_national_performances.csv')
    
    # 1. Load profiles
    print("Loading profiles...")
    df_profiles = pd.read_csv(profiles_path, low_memory=False)
    # We only need specific columns
    df_profiles = df_profiles[['player_id', 'date_of_birth', 'height', 'position']]
    
    # Calculate Age
    current_year = datetime.now().year
    df_profiles['Age'] = current_year - pd.to_datetime(df_profiles['date_of_birth'], errors='coerce').dt.year
    df_profiles.drop('date_of_birth', axis=1, inplace=True)
    
    # Clean height
    df_profiles['height'] = pd.to_numeric(df_profiles['height'], errors='coerce')
    
    # Clean position (take just the main position if it's complex)
    df_profiles['position'] = df_profiles['position'].astype(str).str.split(' - ').str[0]
    
    # 2. Load market value
    print("Loading market values...")
    df_market_value = pd.read_csv(market_value_path)
    df_market_value = df_market_value[['player_id', 'value']]
    df_market_value.rename(columns={'value': 'Market_Value'}, inplace=True)
    
    # Filter out 0.0 values or missing values as we need a valid target
    df_market_value = df_market_value[df_market_value['Market_Value'] > 0]
    
    # 3. Load national performances
    print("Loading national performances...")
    df_national = pd.read_csv(national_perf_path)
    # Group by player_id and sum matches and goals (they might have U21 and Senior teams)
    df_national_agg = df_national.groupby('player_id')[['matches', 'goals']].sum().reset_index()
    df_national_agg.rename(columns={'matches': 'National_Matches', 'goals': 'National_Goals'}, inplace=True)
    
    # 4. Merge all together
    print("Merging datasets...")
    df_merged = pd.merge(df_market_value, df_profiles, on='player_id', how='inner')
    df_merged = pd.merge(df_merged, df_national_agg, on='player_id', how='left')
    
    # Fill missing national matches/goals with 0 (assuming no record means 0)
    df_merged['National_Matches'] = df_merged['National_Matches'].fillna(0)
    df_merged['National_Goals'] = df_merged['National_Goals'].fillna(0)
    
    # Drop player_id as it's not a feature
    df_merged.drop('player_id', axis=1, inplace=True)
    
    # Drop any rows with missing essential features (Age, Height, Position)
    df_merged.dropna(subset=['Age', 'height', 'position'], inplace=True)
    
    # Remove extreme outliers in height if any (e.g., 0 height)
    df_merged = df_merged[(df_merged['height'] > 140) & (df_merged['height'] < 220)]
    
    # Rename columns to match existing pipeline style
    df_merged.rename(columns={
        'height': 'Height_cm',
        'position': 'Position'
    }, inplace=True)
    
    output_path = os.path.join(base_dir, 'raw_data', 'sports_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_merged.to_csv(output_path, index=False)
    
    print(f"Dataset successfully generated at {output_path} with {len(df_merged)} rows.")

if __name__ == '__main__':
    prepare_data()
