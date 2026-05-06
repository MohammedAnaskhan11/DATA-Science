import pandas as pd
import numpy as np
import os

def generate_sports_data(num_samples=1000):
    np.random.seed(42)
    
    # Generate realistic-looking data
    age = np.random.normal(26, 4, num_samples).astype(int)
    matches_played = np.random.randint(10, 82, num_samples)
    minutes_per_match = np.random.normal(25, 8, num_samples).clip(5, 48)
    
    # Let's say training hours and age influence goals/points
    training_hours_per_week = np.random.normal(15, 5, num_samples).clip(5, 30)
    
    # Calculate base points based on minutes and training, with some noise
    points_per_game = (minutes_per_match * 0.4) + (training_hours_per_week * 0.2) + np.random.normal(0, 3, num_samples)
    points_per_game = points_per_game.clip(0, 40)
    
    assists_per_game = (minutes_per_match * 0.15) + np.random.normal(0, 2, num_samples)
    assists_per_game = assists_per_game.clip(0, 15)
    
    # Target variable: Performance Score (0-100)
    # A combination of points, assists, with a slight penalty for older age
    performance_score = (points_per_game * 1.5) + (assists_per_game * 2) + (training_hours_per_week * 0.5) - (age - 25) * 0.5 + np.random.normal(0, 5, num_samples)
    performance_score = (performance_score / performance_score.max()) * 100
    performance_score = performance_score.clip(0, 100)
    
    # Introduce some missing values to simulate real-world data
    df = pd.DataFrame({
        'Age': age,
        'Matches_Played': matches_played,
        'Minutes_Per_Match': minutes_per_match,
        'Training_Hours_Per_Week': training_hours_per_week,
        'Points_Per_Game': points_per_game,
        'Assists_Per_Game': assists_per_game,
        'Performance_Score': performance_score
    })
    
    # Randomly set 2% of Training_Hours and Assists to NaN
    df.loc[df.sample(frac=0.02).index, 'Training_Hours_Per_Week'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'Assists_Per_Game'] = np.nan
    
    return df

if __name__ == '__main__':
    df = generate_sports_data(1500)
    # Define paths relatively
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'raw_data', 'sports_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}")
