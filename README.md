# Credit Risk Prediction — Model Card

A machine learning system that predicts the probability a borrower will experience serious delinquency (90+ days past due) within the next two years, built on the Kaggle "Give Me Some Credit" dataset.

---

## 1. Project Overview

**Task:** Binary classification — predict `SeriousDlqin2yrs` (1 = borrower experiences serious delinquency within two years, 0 = does not).

**Dataset:** Kaggle "Give Me Some Credit" — 150,000 borrowers, 11 selected features after feature engineering and feature selection, ~7% positive class (serious delinquency).

**Approach:** Three gradient-boosted tree models (XGBoost, LightGBM, CatBoost) tuned independently via Optuna, combined into a stacked ensemble with a logistic regression meta-learner trained on out-of-fold predictions.

**Final model:** Stacking ensemble (`meta_model.pkl`), built on top of `xgb_best.pkl`, `lgbm_best.pkl`, and `cat_best.pkl`.

---

## 2. Intended Use

This project is a portfolio and learning exercise demonstrating an end-to-end credit risk modeling workflow: EDA, feature engineering, feature selection, imbalanced-class handling, hyperparameter tuning, ensembling, and explainability.

It is **not** intended for deployment in an actual lending decision system. The dataset is a public Kaggle competition dataset, not real production data, and the model has not undergone the regulatory, fairness, or legal review (e.g. ECOA/Regulation B compliance in the U.S., adverse action notice requirements) that a real credit decisioning model would require before use.

Appropriate uses: demonstrating ML workflow competency, a reference implementation for imbalanced binary classification with tree ensembles, an educational example of model explainability techniques (SHAP, PDP) applied to a financial use case.

Inappropriate uses: making real lending decisions, informing real credit limits or interest rates, any use affecting a real individual's access to credit.

---

## 3. Training Data

**Source:** Kaggle "Give Me Some Credit" competition dataset (`cs-training.csv`), 150,000 rows.

**Target:** `SeriousDlqin2yrs` — binary indicator of serious delinquency within two years. Base rate ~7%, a substantial class imbalance that shaped most downstream modeling decisions.

**Features used in the final model** (11, after feature engineering in Phase 3 and feature selection in Phase 4 via RFE, SHAP importance, and VIF):

- Engineered ratios and interaction terms (e.g. `LatePaymentRatio`, debt-to-income-style ratios)
- `RevolvingUtilizationOfUnsecuredLines`
- `TotalLatePayments`
- `NumberOfDependents`
- `NumberRealEstateLoansOrLines`
- `DebtRatio`
- Additional engineered/binned features retained after VIF and collinearity filtering

Features were dropped if they showed high collinearity (VIF), low RFE-ranked predictive value, or negligible SHAP importance. Every retained feature was checked for multicollinearity given the importance of interpretability in a credit risk context.

**Known data limitations:**
- Several raw fields (e.g. income, dependents) contained missing values, handled during EDA (Phase 2); imputation choices can introduce bias if missingness correlates with the outcome in ways not captured by the imputation strategy.
- The dataset reflects historical lending and repayment patterns from its original collection period and population; it may not generalize to other time periods, geographies, or borrower populations.
- The ~7% positive class rate means even strong models have limited precision at any reasonable recall — this is a fundamental property of the problem, not a modeling shortfall, and is reflected in the PR-AUC and F1 numbers below.

---

## 4. Modeling Approach

**Phase 5 — Baselines:** XGBoost, LightGBM, and CatBoost trained with SMOTE oversampling applied to the training folds only (never the test set), evaluated primarily on PR-AUC rather than accuracy, given the class imbalance.

**Phase 6 — Tuning and stacking:**
- Hyperparameters tuned independently for each of the three models using Optuna (50 trials, TPE sampler, PR-AUC as the optimization objective).
- A 5-fold cross-validation scheme generated out-of-fold (OOF) predictions for each base model on the training set, and averaged test-set predictions across the 5 fold-models.
- SMOTE was applied within each training fold only, never leaking into validation or test folds.
- The three sets of OOF predictions were used as features (`xgb_pred`, `lgbm_pred`, `cat_pred`) to train a logistic regression meta-learner, which was then evaluated on the corresponding averaged test-set predictions.

This OOF-based stacking design is deliberate: it prevents the meta-learner from training on predictions that have already seen the same data point during base-model fitting, which would otherwise leak information and produce overly optimistic stacked performance.

---

## 5. Evaluation Results

All metrics computed on a held-out test set (20% of data, stratified split, `random_state=42`, n=30,000), never used in training or hyperparameter tuning.

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| XGBoost (baseline) | 0.8506 | 0.3383 | 0.3874 |
| LightGBM (baseline) | 0.8505 | 0.3358 | 0.3728 |
| CatBoost (baseline) | 0.8534 | 0.3395 | 0.3976 |
| **Stacking Ensemble (final)** | **0.8588** | **0.3558** | **0.3698** |

**PR-AUC is the primary metric** given the ~7% positive class rate — ROC-AUC can look deceptively strong on imbalanced problems since it's dominated by performance on the majority class, while PR-AUC more directly reflects performance on the minority (default) class that actually matters for this task.

The stacking ensemble improves PR-AUC by roughly 0.016 over the best individual baseline (CatBoost), and improves ROC-AUC by roughly 0.005. The F1 score for the stack (0.3698) is slightly below CatBoost's baseline F1 (0.3976) at the default 0.5 threshold — this is a known characteristic of stacked models with recalibrated probabilities, and a different decision threshold (tuned on the PR curve rather than left at 0.5) would likely close or reverse this gap. This tradeoff is reported here rather than omitted.

---

## 6. Explainability

**Global feature importance (SHAP):** A SHAP summary plot and bar chart (`reports/figures/shap_summary.png`, `reports/figures/shap_bar.png`) were generated on a 5,000-row sample of the test set using `TreeExplainer` on the XGBoost model. `RevolvingUtilizationOfUnsecuredLines` and `LatePaymentRatio`/`TotalLatePayments` rank among the most influential features, consistent with domain expectations for credit risk — utilization and delinquency history are classic risk signals.

**Individual prediction explanations:** Waterfall plots (`reports/figures/waterfall_high_risk.png`, `reports/figures/waterfall_low_risk.png`) show how individual feature values push a single prediction above or below the baseline expected value, illustrating the kind of borrower-level explanation that would be required in a real deployed credit system (e.g. for adverse action notices).

**Partial dependence plots:** PDP plots (`reports/figures/pdp_top_features.png`) for the top 6 features by SHAP importance show how predicted risk changes as each feature varies in isolation, holding others fixed. These reveal whether the model's learned relationships are monotonic or have nonlinear inflection points worth highlighting — for example, a steep rise in predicted risk past a particular utilization threshold rather than a smooth linear increase.

---

## 7. Known Limitations and Bias Considerations

- **Synthetic minority oversampling (SMOTE):** SMOTE generates synthetic examples by interpolating between existing minority-class points. If certain subpopulations are underrepresented in the original delinquent class, SMOTE will not correct for that absence — it can only interpolate within the data it has, not invent genuinely missing patterns.
- **No demographic fairness audit:** This project does not include an analysis of model performance or error rates broken out by protected characteristics (race, gender, age, etc.), because the dataset does not include these fields. A real-world deployment would require such an audit, since proxy discrimination through correlated features (e.g. ZIP code, certain financial behaviors) is a well-documented risk in credit scoring models.
- **Threshold sensitivity:** All F1 scores in this model card use a default 0.5 classification threshold, which was not tuned for any specific business objective (e.g. minimizing false negatives on defaults vs. minimizing false positives that deny credit to creditworthy applicants). A real deployment would require threshold selection tied to a specific cost-benefit analysis.
- **Temporal validity:** The dataset reflects a specific historical period. Credit behavior, macroeconomic conditions, and lending practices change over time, and the model has not been validated against more recent data or tested for performance drift.
- **Base-model identity in stacking:** The out-of-fold base predictions used to train the meta-learner come from 5 fold-specific models per algorithm (not the single final `*_best.pkl` models saved for inference), averaged across folds for the test set. This is standard stacking practice but means the meta-learner's training signal differs slightly in distribution from predictions generated by the single saved base models at actual inference time — a known and generally small source of train/serve skew.

---

## 8. Project Structure

```
credit-risk/
├── configs/
│   └── best_params.json          # Optuna-tuned hyperparameters (xgb, lgbm, cat)
├── data/
│   ├── raw/
│   │   └── cs-training.csv
│   └── processed/
│       ├── df_features.csv       # engineered features
│       ├── df_log_features.csv   # log-transformed features
│       ├── df_selected.csv       # post feature-selection
│       ├── X_stack_train.csv     # OOF base-model predictions (meta-learner training)
│       ├── X_stack_test.csv      # averaged base-model predictions (meta-learner test)
│       ├── oof_arrays.npz        # raw OOF prediction arrays
│       └── baseline_results.csv  # Phase 5 baseline metrics
├── models/
│   ├── xgb_best.pkl
│   ├── lgbm_best.pkl
│   ├── cat_best.pkl
│   └── meta_model.pkl            # stacking meta-learner
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_hpo_ensemble.ipynb
│   └── 06_explainability.ipynb
├── reports/
│   └── figures/                  # EDA plots, PR curves, SHAP/PDP plots
├── src/
│   ├── features.py               # feature engineering pipeline
│   └── train.py                  # training pipeline
└── README.md                     # this file
```

---

## 9. How to Reproduce

1. Place `cs-training.csv` in `data/raw/`.
2. Run notebooks in order, 01 through 06. Each notebook reads from and writes to `data/processed/` and persists artifacts (`models/`, `configs/`, `reports/figures/`) used by subsequent notebooks.
3. Final model artifacts (`models/*.pkl`) and tuned hyperparameters (`configs/best_params.json`) are produced by `05_hpo_ensemble.ipynb`.
4. `06_explainability.ipynb` loads the saved models and produces the SHAP/PDP figures and the metrics in Section 5 of this model card.
