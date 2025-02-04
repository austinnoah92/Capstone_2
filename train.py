import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import xgboost

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from scipy.interpolate import PchipInterpolator

# Import your feature engineering function
from feature_engineering import feature_engineering

#############################################
# Spline Calibration Class
#############################################
class SplineCalibration:
    def __init__(self):
        self.spline = None
    
    def fit(self, s, y):
        # Sort raw probabilities and corresponding true labels
        sorted_idx = np.argsort(s)
        s_sorted = s[sorted_idx]
        y_sorted = y[sorted_idx]
        # Get unique s values and compute empirical positive rate for each unique value
        s_unique, idx, counts = np.unique(s_sorted, return_index=True, return_counts=True)
        empirical = np.array([np.mean(y_sorted[i:i+c]) for i, c in zip(idx, counts)])
        self.spline = PchipInterpolator(s_unique, empirical)
        return self
    
    def predict_proba(self, s):
        calibrated = self.spline(s)
        return np.clip(calibrated, 0, 1)

#############################################
# Compute Optimal Threshold Function
#############################################
def compute_optimal_threshold(model, X, y):
    
    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    y_true_all, y_pred_all = [], []
    
    for train_idx, val_idx in tqdm(skfold.split(X, y), total=skfold.get_n_splits(), desc="CV Folds"):
        df_train = X.iloc[train_idx]
        df_val   = X.iloc[val_idx]
        y_train = y.iloc[train_idx].values
        y_val   = y.iloc[val_idx].values
        
        dv = DictVectorizer(sparse=False)
        X_train = dv.fit_transform(df_train.to_dict(orient='records'))
        X_val   = dv.transform(df_val.to_dict(orient='records'))
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        y_true_all.extend(y_val)
        y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    # Apply Spline Calibration
    spline_cal = SplineCalibration().fit(y_pred_all, y_true_all)
    y_pred_calibrated = spline_cal.predict_proba(y_pred_all)
    
    precision, recall, thresholds = precision_recall_curve(y_true_all, y_pred_calibrated)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # The first point is implicit; skip it.
    optimal_idx = np.argmax(f1_scores[1:])
    optimal_threshold = thresholds[optimal_idx]
    
    pr_auc = average_precision_score(y_true_all, y_pred_calibrated)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC={pr_auc:.2f})')
    plt.scatter(recall[optimal_idx+1], precision[optimal_idx+1], color='red', marker='o', 
                label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Spline Calibrated)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Optimal Threshold (Spline Calibrated): {optimal_threshold:.2f}")
    print(f"PR-AUC (Spline Calibrated): {pr_auc:.4f}")
    
    return optimal_threshold

#############################################
# Train Final Model Function
#############################################
def train_final_model(model, X, y):
    """
    Trains the provided model on the full training data after one-hot encoding using DictVectorizer.
    Returns the trained model and the fitted DictVectorizer.
    """
    dv = DictVectorizer(sparse=False)
    X_transformed = dv.fit_transform(X.to_dict(orient='records'))
    model.fit(X_transformed, y)
    return model, dv

#############################################
# Evaluate Model Function
#############################################
def evaluate_model(model, dv, X, y, threshold):
    """
    Transforms the data using the provided DictVectorizer, obtains predicted probabilities,
    applies the threshold, and returns evaluation metrics.
    """
    X_transformed = dv.transform(X.to_dict(orient='records'))
    proba = model.predict_proba(X_transformed)[:, 1]
    predictions = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1_score": f1_score(y, predictions)
    }
    return metrics

#############################################
# Main Training Script
#############################################
if __name__ == '__main__':
    # Load your training data (adjust path as needed)
    train_df = pd.read_csv("C:/Users/austi/Videos/Capstone_2/Train.csv")
    del train_df['Total_Income']
    train_df = feature_engineering(train_df)
    
    # Split into features and target (assume target is 'Loan_Status')
    features = [col for col in train_df.columns if col != 'Loan_Status']
    X_full_train = train_df[features]
    y_full_train = train_df['Loan_Status']
    
    # Optionally, split a hold-out test set from training data (or use a separate test set)
    X_full_train, X_val, y_full_train, y_val = train_test_split(X_full_train, y_full_train, test_size=0.20, random_state=42, stratify=y_full_train)
    
    # Compute optimal thresholds for each model using Spline Calibration via CV.
    optimal_thresh_lr = compute_optimal_threshold(
        LogisticRegression(solver='liblinear', penalty='l1', C=0.01, max_iter=1000),
        X_full_train, y_full_train)
    
    optimal_thresh_rf = compute_optimal_threshold(
        RandomForestClassifier(random_state=1),
        X_full_train, y_full_train)
    
    optimal_thresh_dt = compute_optimal_threshold(
        DecisionTreeClassifier(random_state=1),
        X_full_train, y_full_train)
    
    optimal_thresh_gb = compute_optimal_threshold(
        GradientBoostingClassifier(random_state=1),
        X_full_train, y_full_train)
    
    optimal_thresh_xgb = compute_optimal_threshold(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1),
        X_full_train, y_full_train)
    
    print("Optimal Thresholds (Spline Calibrated):")
    print(f"Logistic Regression: {optimal_thresh_lr:.2f}")
    print(f"Random Forest:       {optimal_thresh_rf:.2f}")
    print(f"Decision Tree:       {optimal_thresh_dt:.2f}")
    print(f"Gradient Boosting:   {optimal_thresh_gb:.2f}")
    print(f"XGBoost:             {optimal_thresh_xgb:.2f}")
    
    # Retrain final models on the full training data
    model_lr_final, dv_lr_final = train_final_model(
        LogisticRegression(solver='liblinear', penalty='l1', C=0.01, max_iter=1000),
        X_full_train, y_full_train)
    
    model_rf_final, dv_rf_final = train_final_model(
        RandomForestClassifier(random_state=1),
        X_full_train, y_full_train)
    
    model_dt_final, dv_dt_final = train_final_model(
        DecisionTreeClassifier(random_state=1),
        X_full_train, y_full_train)
    
    model_gb_final, dv_gb_final = train_final_model(
        GradientBoostingClassifier(random_state=1),
        X_full_train, y_full_train)
    
    model_xgb_final, dv_xgb_final = train_final_model(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1),
        X_full_train, y_full_train)
    
    # Evaluate final models on the test set
    print("\nEvaluation Metrics on Test Set:")
    print("=" * 40)
    models_info = {
        "Logistic Regression": (model_lr_final, dv_lr_final, optimal_thresh_lr),
        "Random Forest": (model_rf_final, dv_rf_final, optimal_thresh_rf),
        "Decision Tree": (model_dt_final, dv_dt_final, optimal_thresh_dt),
        "Gradient Boosting": (model_gb_final, dv_gb_final, optimal_thresh_gb),
        "XGBoost": (model_xgb_final, dv_xgb_final, optimal_thresh_xgb)
    }
    
    for model_name, (model, dv, threshold) in models_info.items():
        metrics = evaluate_model(model, dv, X_val, y_val, threshold)
        print(f"{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        print("-" * 40)
    
    ##########################################
    # Save the best model artifacts with joblib
    ##########################################
    
    joblib.dump(model_lr_final, 'best_model.pkl')
    joblib.dump(dv_lr_final, 'best_dv.pkl')
    joblib.dump(optimal_thresh_lr, 'optimal_threshold.pkl')
    
    print("\nBest model artifacts saved as 'best_model.pkl', 'best_dv.pkl', and 'optimal_threshold.pkl'.")
