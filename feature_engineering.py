import pandas as pd
import numpy as np


def feature_engineering(df):
    # 1. Combine Income Features
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

    # 2. Loan-to-Income Ratio
    df['Loan_to_Income_Ratio'] = df['LoanAmount'] / df['Total_Income']

    # 3. Loan_Term_Category:
    # If Loan_Amount_Term is constant, assign a default category; otherwise, bin it.
    if df['Loan_Amount_Term'].nunique() == 1:
        df['Loan_Term_Category'] = 'Constant'
    else:
        df['Loan_Term_Category'] = pd.qcut(
            df['Loan_Amount_Term'], 
            q=3, 
            labels=['Short-Term', 'Medium-Term', 'Long-Term'], 
            duplicates='drop'
        )

    # 4. Interaction between Married and Education
    df['Married_Education_Interaction'] = df['Married'] * df['Education']

    # 5. Categorize Married-Education Interaction
    conditions = [
        (df['Married'] == 1) & (df['Education'] == 1),
        (df['Married'] == 1) & (df['Education'] == 0),
        (df['Married'] == 0) & (df['Education'] == 1),
        (df['Married'] == 0) & (df['Education'] == 0)
    ]
    choices = ['Married_Graduate', 'Married_NonGraduate', 'NotMarried_Graduate', 'NotMarried_NonGraduate']
    df['Married_Education_Category'] = np.select(conditions, choices, default='Unknown')

    # 6. Log Transformations
    df['Log_LoanAmount'] = np.log(df['LoanAmount'] + 1)
    df['Log_ApplicantIncome'] = np.log(df['ApplicantIncome'] + 1)
    df['Log_CoapplicantIncome'] = np.log(df['CoapplicantIncome'] + 1)
    df['Log_Total_Income'] = np.log(df['Total_Income'] + 1)

    # 7. Drop redundant columns
    drop_cols = ['LoanAmount', 'ApplicantIncome', 'CoapplicantIncome', 'Total_Income', 'Education', 'Loan_ID']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df
