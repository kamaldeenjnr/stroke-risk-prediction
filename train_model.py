"""
train_model.py
----------------
End-to-end training pipeline for the Stroke Prediction dataset.

Usage:
    python train_model.py --data data/healthcare-dataset-stroke-data.csv

What this does (in order):
  1. Load + clean the raw CSV (fixes missing BMI values, drops id column)
  2. Build a preprocessing pipeline (one-hot encode categoricals, scale numerics)
  3. Split into train/test (stratified, 80/20) -- important given ~5% positive rate
  4. Balance the TRAINING data only with SMOTE (no leakage into test set;
     test set keeps the real-world class imbalance so evaluation is honest)
  5. Train + hyperparameter-tune three model families:
       - Logistic Regression (baseline, interpretable)
       - Random Forest
       - XGBoost
  6. Select the best model by cross-validated ROC-AUC (not accuracy --
     accuracy is a misleading metric on this imbalanced a dataset)
  7. Evaluate the best model on the held-out test set, reporting recall
     prominently since false negatives (missed stroke risk) matter most
  8. Save the trained pipeline + all evaluation artifacts
"""

import argparse
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. DATA LOADING + CLEANING
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # id is a pure identifier -- drop it, it carries no signal
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # bmi has missing values (sometimes stored as "N/A" strings, sometimes NaN).
    # Coerce to numeric and leave NaN in place -- the pipeline's imputer handles it.
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Drop the single "Other" gender row if present (dataset has 1 or 2 -- too
    # few to encode meaningfully, and it's usually a single outlier record)
    if "gender" in df.columns:
        df = df[df["gender"].isin(["Male", "Female"])].copy()

    # Target is already 0/1 in this dataset, but make sure it's int
    df["stroke"] = df["stroke"].astype(int)

    return df


# ---------------------------------------------------------------------------
# 2. PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame, target_col: str = "stroke"):
    feature_df = df.drop(columns=[target_col])

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# 3. QUICK EDA PLOTS
# ---------------------------------------------------------------------------
def save_eda_plots(df: pd.DataFrame, outdir: str) -> None:
    sns.set_style("whitegrid")

    # Stroke rate by age group
    plt.figure(figsize=(7, 5))
    age_bins = pd.cut(df["age"], bins=[0, 20, 40, 60, 80, 100])
    stroke_by_age = df.groupby(age_bins, observed=True)["stroke"].mean()
    stroke_by_age.plot(kind="bar", color="#C44E52")
    plt.ylabel("Stroke Rate")
    plt.xlabel("Age Group")
    plt.title("Stroke Rate by Age Group")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "stroke_rate_by_age_group.png"), dpi=150)
    plt.close()

    # Glucose level vs stroke
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="stroke", y="avg_glucose_level", palette=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["No Stroke", "Stroke"])
    plt.title("Average Glucose Level by Stroke Outcome")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "glucose_vs_stroke.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4. MODEL TRAINING + HYPERPARAMETER SEARCH
# ---------------------------------------------------------------------------
def get_candidate_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 400, 600],
                "classifier__max_depth": [None, 8, 12, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [200, 400, 600],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__subsample": [0.7, 0.8, 1.0],
                "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            },
        ),
    }


def train_and_select_best(X_train, y_train, preprocessor, cv_folds=5, n_iter=15):
    candidates = get_candidate_models()
    results = {}
    fitted_searches = {}

    for name, (estimator, param_dist) in candidates.items():
        print(f"\n--- Tuning {name} ---")
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("classifier", estimator),
            ]
        )

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_folds,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        print(f"{name} best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"{name} best params: {search.best_params_}")

        results[name] = search.best_score_
        fitted_searches[name] = search

    best_name = max(results, key=results.get)
    print(f"\n=== Best model: {best_name} (CV ROC-AUC = {results[best_name]:.4f}) ===")
    return best_name, fitted_searches[best_name].best_estimator_, results


# ---------------------------------------------------------------------------
# 5. EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, outdir: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Test Set Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nNote: recall is the metric that matters most here -- it tells you")
    print("what fraction of actual stroke cases the model successfully flagged.")
    print("Accuracy alone is misleading on a ~5% positive-rate dataset.")

    report = classification_report(y_test, y_pred, target_names=["No Stroke", "Stroke"])
    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Stroke", "Stroke"])
    disp.plot(cmap="Reds", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close()

    return metrics


def save_feature_importance(model, numeric_cols, categorical_cols, outdir: str) -> None:
    classifier = model.named_steps["classifier"]

    fitted_preprocessor = model.named_steps["preprocessor"]
    cat_pipeline = fitted_preprocessor.named_transformers_["cat"]
    cat_encoder = cat_pipeline.named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    if hasattr(classifier, "feature_importances_"):
        # Tree-based models (Random Forest, XGBoost)
        importances = classifier.feature_importances_
        imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, x="importance", y="feature", color="#C44E52")
        plt.title("Top 15 Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
        plt.close()

    elif hasattr(classifier, "coef_"):
        # Linear models (Logistic Regression) -- use signed coefficient
        # magnitude as a proxy for importance. Sign matters: positive
        # coefficients push toward "stroke", negative push toward "no stroke".
        coefs = classifier.coef_[0]
        imp_df = pd.DataFrame({"feature": all_feature_names, "importance": coefs})
        imp_df["abs_importance"] = imp_df["importance"].abs()
        imp_df = imp_df.sort_values("abs_importance", ascending=False).head(15)
        imp_df["direction"] = imp_df["importance"].apply(lambda x: "Increases risk" if x > 0 else "Decreases risk")

        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=imp_df, x="importance", y="feature",
            hue="direction", palette={"Increases risk": "#C44E52", "Decreases risk": "#4C72B0"},
            dodge=False,
        )
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Top 15 Logistic Regression Coefficients\n(magnitude = influence, color = direction)")
        plt.xlabel("Coefficient (standardized features)")
        plt.legend(title="")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "feature_importance.png"), dpi=150)
        plt.close()

    else:
        print("Selected model has neither feature_importances_ nor coef_; skipping importance plot.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a stroke risk prediction model.")
    parser.add_argument("--data", type=str, required=True, help="Path to healthcare-dataset-stroke-data.csv")
    parser.add_argument("--outdir", type=str, default="outputs", help="Where to save plots/metrics")
    parser.add_argument("--modeldir", type=str, default="models", help="Where to save the trained model")
    parser.add_argument("--n_iter", type=int, default=15, help="RandomizedSearchCV iterations per model")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean(args.data)
    print(f"Loaded {len(df)} rows, stroke rate = {df['stroke'].mean():.2%}")

    print("Saving EDA plots...")
    save_eda_plots(df, args.outdir)

    print("Building preprocessing pipeline...")
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df)

    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train stroke rate: {y_train.mean():.2%}, Test stroke rate: {y_test.mean():.2%}")

    best_name, best_model, cv_results = train_and_select_best(
        X_train, y_train, preprocessor, n_iter=args.n_iter
    )

    metrics = evaluate_model(best_model, X_test, y_test, args.outdir)
    metrics["best_model"] = best_name
    metrics["cv_roc_auc_by_model"] = cv_results

    save_feature_importance(best_model, numeric_cols, categorical_cols, args.outdir)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.modeldir, "best_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nSaved trained pipeline to {model_path}")
    print(f"Saved metrics + plots to {args.outdir}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
