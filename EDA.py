# EDA.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def check_missing_values(df):
    
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing Percentage': percent})
    print("Missing Values:\n", missing_df)
    
def check_duplicates(df):
    
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
def plot_numerical_distributions(df):
    
    num_df = df.select_dtypes(include=['float64', 'int64'])
    for col in num_df.columns:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(num_df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=num_df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()
        
def plot_categorical_distributions(df):
   
    cat_df = df.select_dtypes(include=['object'])
    for col in cat_df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=cat_df[col])
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
def plot_correlation_heatmap(df):
    
    num_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

def plot_target_distribution(df, target='Loan_Status'):
    
    plt.figure(figsize=(6, 4))
    df[target].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {target}')
    plt.xlabel(target)
    plt.ylabel('Count')
    plt.show()

def plot_feature_vs_target(df, target='Loan_Status'):
    
    for col in df.columns:
        if col == target:
            continue
        if df[col].dtype in ['int64', 'float64']:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f'{col} vs {target}')
            plt.show()
        else:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=col, hue=target, data=df)
            plt.title(f'{col} vs {target}')
            plt.xticks(rotation=45)
            plt.show()

def perform_eda(df, target='Loan_Status'):
    
    print("----- Data Info -----")
    print(df.info())
    print("\n----- Summary Statistics -----")
    print(df.describe())
    print("\n----- Missing Values -----")
    check_missing_values(df)
    print("\n----- Duplicate Rows -----")
    check_duplicates(df)
    print("\n----- Numerical Distributions -----")
    plot_numerical_distributions(df)
    print("\n----- Categorical Distributions -----")
    plot_categorical_distributions(df)
    print("\n----- Correlation Heatmap -----")
    plot_correlation_heatmap(df)
    print("\n----- Target Distribution -----")
    plot_target_distribution(df, target)
    print("\n----- Feature vs Target -----")
    plot_feature_vs_target(df, target)
    
if __name__ == '__main__':
    # Example usage: load your training data and perform EDA.
    import sys
    # Adjust file path as needed.
    train = pd.read_csv("C:/Users/austi/Videos/Capstone_2/Train.csv")
    perform_eda(train)
