# Loan Default Prediction - Hackathon Submission

## Project Overview
This project predicts loan defaults using advanced machine learning techniques on a highly imbalanced dataset (5.1% default rate). The final stacking ensemble achieves **AUC 0.8074** on validation data.

## Quick Start
1. Run notebooks in numerical order: `1cleaning_a.ipynb` → `2eda.ipynb` → `3feature_e.ipynb` → `4modeling_logistic.ipynb` → `5modeling_lightgbm.ipynb`
2. For evaluation predictions: run `evaluation_set/model.ipynb`
3. Results saved to `evaluation_set/results.csv`

## Project Structure
```
├── 1cleaning_a.ipynb          # Data cleaning and preprocessing
├── 2eda.ipynb                  # Exploratory data analysis
├── 3feature_e.ipynb            # Feature engineering (111 features)
├── 4modeling_logistic.ipynb   # Baseline logistic regression
├── 5modeling_lightgbm.ipynb   # Advanced models and ensembles
├── evaluation_set/
│   ├── model.ipynb            # Evaluation predictions
│   └── results.csv            # Final predictions (10,001 rows)
├── final_lgbm_optimized.pkl   # Best model (stacking ensemble)
└── dataset_with_features.csv  # Processed data with features
```


## Model Performance
| Model | AUC | Precision | Recall | F1-Score | Overfit Gap |
|-------|-----|-----------|--------|----------|-------------|
| Logistic Regression | 0.7927 | 0.4157 | 0.2862 | 0.3390 | 0.1567 |
| XGBoost | 0.7998 | 0.4184 | 0.2988 | 0.3485 | 0.1792 |
| LightGBM Baseline | 0.7941 | 0.4212 | 0.2819 | 0.3382 | 0.1567 |
| LightGBM Optimized | 0.8044 | 0.4212 | 0.2884 | 0.3419 | 0.0278 |
| **Stacking Ensemble** | **0.8074** | **0.4202** | **0.2884** | **0.3411** | **0.0264** |

## Key Techniques
- **Feature Engineering**: 17 custom features (geographic, financial ratios, risk scores)
- **Target Encoding**: Mean encoding with k-fold smoothing
- **Handling Imbalance**: Percentile-based threshold (95th percentile = 0.212)
- **Ensembling**: Stacking with XGBoost, LightGBM, and Logistic meta-learner

## Critical Implementation Details
- **Threshold Selection**: Used 95th percentile of predicted probabilities to match training default rate (5.1%)
- **Feature Count**: 111 total features (17 engineered + one-hot encoding)
- **Training Time**: ~5 minutes for full pipeline on standard laptop

## Evaluation Results
- **Total Predictions**: 10,001
- **Predicted Defaults**: 511 (5.11%)
- **Output Format**: CSV with columns [loan_id, default]
- **File Location**: `evaluation_set/results.csv`

## Dependencies
```
pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, seaborn, imbalanced-learn
```

