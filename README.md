# SportsData Insight: Player Market Value Prediction

## Project Details
**Domain**: Sports Data Science
**SDG Goal**: Goal 3: Health and Well-being

## Abstract
This project aims to apply Data Science techniques to analyze real-world sports data and predict player market value. Sports organizations heavily rely on performance analysis and market valuation to improve their recruitment strategies. The project involves data collection from Transfermarkt, preprocessing, exploratory data analysis, visualization, and model development to derive actionable insights for predicting player market value based on demographics and national team performances.

## Problem Statement
Football clubs need robust mechanisms to evaluate players and make informed financial decisions in the transfer market. This project analyzes real-world historical player data to build a predictive model for player market value.

## Dataset Source
*   The dataset is derived from the **Transfermarkt** datalake. It contains features such as Age, Height, Position, National Team Matches, and National Team Goals. 
*   A processing script (`dataset/prepare_real_data.py`) joins player profiles, market values, and national team performances to output the final training dataset located at `dataset/raw_data/sports_data.csv`.

## Methodology / Workflow
1.  **Problem Identification**: Defining the objectives and scope of market value prediction.
2.  **Dataset Processing**: Extracting, aggregating, and joining data from Transfermarkt CSV files.
3.  **Data Cleaning / Preprocessing**: Handled missing values (imputed with medians/modes), performed one-hot encoding for categorical variables (Position), and scaled features using `StandardScaler`.
4.  **Exploratory Data Analysis (EDA)**: Analyzed feature distributions and identified top correlations (e.g., National Team Matches and Goals) with Market Value.
5.  **Data Visualization**: Generated correlation heatmaps and scatter plots (found in `outputs/graphs/`).
6.  **Model Development**: Trained a **Random Forest Regressor** to predict the Market Value based on player stats.
7.  **Result Interpretation**: Evaluated the model and extracted key metrics.

## Tools Used
*   **Programming Language**: Python
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: Scikit-learn
*   **Visualization**: Matplotlib, Seaborn

## Results / Findings
*   **EDA Findings**: National Team Matches and National Team Goals are the most strongly correlated features with a player's Market Value among the dataset provided.
*   **Model Performance**: The Random Forest Regressor provides an initial baseline for valuation, demonstrating the complex, non-linear relationship between basic player attributes and their financial worth.
*   **Results**: Key metrics such as R-squared and Mean Squared Error are generated dynamically and saved in `outputs/results/model_metrics.txt`.

## Project Structure
```
MiniProject_DS_AIME-B_2024_SportsDataInsight/
├── main.py                          # Entry point – runs the full pipeline
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── generate_data.py             # Synthetic data generator
│   ├── prepare_real_data.py         # Joins Transfermarkt CSVs into training data
│   ├── raw_data/
│   │   └── sports_data.csv          # Raw merged dataset
│   └── processed_data/
│       └── cleaned_sports_data.csv  # Cleaned & preprocessed dataset
│
├── src/
│   ├── preprocessing.py             # Data cleaning, encoding & scaling
│   ├── analysis.py                  # EDA, correlation analysis & visualization
│   └── model.py                     # Random Forest model training & evaluation
│
├── notebooks/
│   ├── data_understanding.ipynb     # Exploratory data analysis notebook
│   ├── preprocessing.ipynb          # Preprocessing walkthrough
│   └── visualization.ipynb          # Visualization experiments
│
├── outputs/
│   ├── graphs/                      # Generated plots (heatmaps, scatter, etc.)
│   │   ├── age_histogram.png
│   │   ├── age_vs_market_value.png
│   │   ├── assists_vs_performance.png
│   │   ├── correlation_heatmap.png
│   │   ├── market_value_boxplot.png
│   │   ├── national_matches_vs_market_value.png
│   │   ├── points_vs_performance.png
│   │   └── top_correlations_barchart.png
│   └── results/
│       └── model_metrics.txt        # R², MSE and other evaluation metrics
│
├── docs/                            # Additional documentation
└── report/                          # Final project report
```

## Team Members
*   **Mohammed Anaskhan S** (Team Leader)
*   **Joe Austin Martin**
