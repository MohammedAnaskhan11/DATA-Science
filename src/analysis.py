import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir=None):
    """
    Performs Exploratory Data Analysis, prints summary stats, 
    and optionally saves visualizations to a directory.
    """
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()
        
        # 2. Scatter Plot: Age vs Market Value
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Age', y='Market_Value', data=df)
        plt.title('Age vs Market Value')
        plt.savefig(os.path.join(output_dir, 'age_vs_market_value.png'))
        plt.close()
        
        # 3. Scatter Plot: National Matches vs Market Value
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='National_Matches', y='Market_Value', data=df)
        plt.title('National Matches vs Market Value')
        plt.savefig(os.path.join(output_dir, 'national_matches_vs_market_value.png'))
        plt.close()
        
        # 4. Histogram: Distribution of Age
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='Age', bins=20, kde=True)
        plt.title('Distribution of Age')
        plt.savefig(os.path.join(output_dir, 'age_histogram.png'))
        plt.close()
        
        # 5. Box Plot: Market Value
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df['Market_Value'])
        plt.title('Box Plot of Market Value')
        plt.xscale('log') # Log scale is better for market values
        plt.savefig(os.path.join(output_dir, 'market_value_boxplot.png'))
        plt.close()
        
        # 6. Bar Chart: Top 10 Correlations with Market Value
        plt.figure(figsize=(10, 6))
        corr = df.corr()['Market_Value'].sort_values(ascending=False).drop('Market_Value')
        top_corr = corr.head(10)
        sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, legend=False)
        plt.title('Top 10 Feature Correlations with Market Value')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_correlations_barchart.png'))
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")

def get_top_correlations(df, target_col):
    """Returns features most correlated with the target column."""
    correlations = df.corr()[target_col].sort_values(ascending=False)
    return correlations.drop(target_col)
