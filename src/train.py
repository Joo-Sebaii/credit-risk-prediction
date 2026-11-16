
import numpy as np
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT   = "/content/drive/MyDrive/credit-risk"
DATA_PROCESSED = f"{PROJECT_ROOT}/data/processed"
MODELS_PATH    = f"{PROJECT_ROOT}/models"
CONFIGS_PATH   = f"{PROJECT_ROOT}/configs"


def load_best_params():
    with open(f"{CONFIGS_PATH}/best_params.json", "r") as f:
        return json.load(f)


def build_models(params):
    xgb = XGBClassifier(**params["xgb"], eval_metric="aucpr", random_state=42, verbosity=0)
    lgbm = LGBMClassifier(**params["lgbm"], random_state=42, verbosity=-1)
    cat = CatBoostClassifier(**params["cat"], random_state=42, verbose=0)
    return xgb, lgbm, cat


def generate_oof_predictions(X_train, y_train, X_test, xgb, lgbm, cat, n_splits=5):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=5)

    n_train = len(X_train)
    oof_xgb  = np.zeros(n_train)
    oof_lgbm = np.zeros(n_train)
    oof_cat  = np.zeros(n_train)
    test_xgb  = np.zeros(len(X_test))
    test_lgbm = np.zeros(len(X_test))
    test_cat  = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train), start=1):
        print(f"  Fold {fold}/{n_splits}")
        X_tr, y_tr   = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

        for model, oof_arr, test_arr in [
            (xgb,  oof_xgb,  test_xgb),
            (lgbm, oof_lgbm, test_lgbm),
            (cat,  oof_cat,  test_cat),
        ]:
            model.fit(X_tr_sm, y_tr_sm)
            oof_arr[valid_idx] = model.predict_proba(X_val)[:, 1]
            test_arr          += model.predict_proba(X_test)[:, 1] / n_splits

    return oof_xgb, oof_lgbm, oof_cat, test_xgb, test_lgbm, test_cat


def train_meta_learner(oof_xgb, oof_lgbm, oof_cat, y_train):
    X_stack_train = pd.DataFrame({
        "xgb_pred":  oof_xgb,
        "lgbm_pred": oof_lgbm,
        "cat_pred":  oof_cat,
    })
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(X_stack_train, y_train)
    return meta, X_stack_train


def evaluate(name, y_test, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    print(f"\n{'='*40}\n  {name}\n{'='*40}")
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"  PR-AUC  : {average_precision_score(y_test, y_proba):.4f}  ← primary")
    print(f"  F1      : {f1_score(y_test, y_pred):.4f}")


def save_artifacts(xgb, lgbm, cat, meta, X_stack_train, X_stack_test,
                   oof_xgb, oof_lgbm, oof_cat):
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(DATA_PROCESSED, exist_ok=True)

    joblib.dump(xgb,  f"{MODELS_PATH}/xgb_best.pkl")
    joblib.dump(lgbm, f"{MODELS_PATH}/lgbm_best.pkl")
    joblib.dump(cat,  f"{MODELS_PATH}/cat_best.pkl")
    joblib.dump(meta, f"{MODELS_PATH}/meta_model.pkl")

    X_stack_train.to_csv(f"{DATA_PROCESSED}/X_stack_train.csv", index=False)
    X_stack_test.to_csv(f"{DATA_PROCESSED}/X_stack_test.csv",  index=False)

    np.savez(f"{DATA_PROCESSED}/oof_arrays.npz",
             oof_xgb=oof_xgb, oof_lgbm=oof_lgbm, oof_cat=oof_cat)

    print("✅ All artifacts saved")


def main():
    df = pd.read_csv(f"{DATA_PROCESSED}/df_selected.csv")
    X  = df.drop(columns=["SeriousDlqin2yrs"])
    y  = df["SeriousDlqin2yrs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = load_best_params()
    xgb, lgbm, cat = build_models(params)

    print("\n🔄 Generating OOF predictions...")
    oof_xgb, oof_lgbm, oof_cat, test_xgb, test_lgbm, test_cat = \
        generate_oof_predictions(X_train, y_train, X_test, xgb, lgbm, cat)

    meta, X_stack_train = train_meta_learner(oof_xgb, oof_lgbm, oof_cat, y_train)

    X_stack_test = pd.DataFrame({
        "xgb_pred":  test_xgb,
        "lgbm_pred": test_lgbm,
        "cat_pred":  test_cat,
    })

    final_proba = meta.predict_proba(X_stack_test)[:, 1]
    evaluate("Stacking Ensemble", y_test, final_proba)

    save_artifacts(xgb, lgbm, cat, meta, X_stack_train, X_stack_test,
                   oof_xgb, oof_lgbm, oof_cat)


if __name__ == "__main__":
    main()
