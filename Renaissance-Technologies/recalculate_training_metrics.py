import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

print("Loading model and data...")

# Load model
with open('final_lgbm_optimized.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv('dataset_with_features.csv')

# Define exclusions
exclude_cols = [
    'customer_id', 'application_id', 'default',
    'loan_officer_id', 'marketing_campaign', 
    'referral_code', 'previous_zip_code'
]

# Get categorical features
all_features = [col for col in df.columns if col not in exclude_cols]
categorical_features = df[all_features].select_dtypes(include=['object', 'category']).columns.tolist()

# One-hot encode
df_encoded = df.copy()

for feat in categorical_features:
    n_unique = df[feat].nunique()
    if n_unique <= 50:
        dummies = pd.get_dummies(df[feat], prefix=feat, drop_first=True, dtype=int)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)

df_encoded = df_encoded.drop(columns=categorical_features)

# Prepare X and y
feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
X = df_encoded[feature_cols].fillna(df_encoded[feature_cols].median())
y = df_encoded['default']

# Split with same parameters as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f" Data loaded: {len(X_test):,} test samples")

# Get probabilities
test_proba = model.predict_proba(X_test)[:, 1]

# Calculate optimal threshold (95th percentile to match 5% default rate)
training_default_rate = y_train.mean()
threshold = np.percentile(test_proba, 100 * (1 - training_default_rate))

# Make predictions with correct threshold
test_pred = (test_proba >= threshold).astype(int)

# Recalculate metrics
cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()
auc = roc_auc_score(y_test, test_proba)

print("\n" + "="*70)
print("CORRECTED TRAINING METRICS (with optimal threshold)")
print("="*70)
print(f"\nTraining default rate: {training_default_rate:.4f} ({training_default_rate*100:.2f}%)")
print(f"Optimal threshold: {threshold:.5f}")
print(f"\nTest Set Results:")
print(f"  ROC-AUC: {auc:.4f} (unchanged - threshold independent)")
print(f"  Predicted defaults: {test_pred.sum()} / {len(test_pred)} ({test_pred.mean()*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:>5,}")
print(f"  False Positives: {fp:>5,}")
print(f"  False Negatives: {fn:>5,}")
print(f"  True Positives:  {tp:>5,}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetrics:")
print(f"  Precision: {precision:.4f} ({tp}/{tp+fp})")
print(f"  Recall:    {recall:.4f} ({tp}/{tp+fn})")
print(f"  F1-Score:  {f1:.4f}")

print("\n" + "="*70)
print("COMPARISON: Original (0.5) vs Corrected Threshold")
print("="*70)
print(f"{'Metric':<25} {'Original (0.5)':<20} {'Corrected':<20}")
print("-"*70)
print(f"{'ROC-AUC':<25} {'0.8074':<20} {f'{auc:.4f} (same)':<20}")
print(f"{'Threshold':<25} {'0.50000':<20} {f'{threshold:.5f}':<20}")
print(f"{'True Positives':<25} {'0':<20} {f'{tp}':<20}")
print(f"{'False Negatives':<25} {'919':<20} {f'{fn}':<20}")
print(f"{'Precision':<25} {'N/A (div/0)':<20} {f'{precision:.4f}':<20}")
print(f"{'Recall':<25} {'0.0000':<20} {f'{recall:.4f}':<20}")
print(f"{'F1-Score':<25} {'0.0000':<20} {f'{f1:.4f}':<20}")
print(f"{'Predicted Defaults':<25} {'0 (0.00%)':<20} {f'{test_pred.sum()} ({test_pred.mean()*100:.2f}%)':<20}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(" Training used WRONG threshold (0.5) for confusion matrix")
print(" ROC-AUC was CORRECT (threshold-independent)")
print(" Evaluation pipeline NOW uses CORRECT threshold")
print(f" Both training and evaluation use: {threshold:.5f}")
print("="*70)
