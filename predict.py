"""
predict.py
----------
Score new patient records with a trained stroke-risk pipeline.

Usage:
    python predict.py --model models/best_model.joblib \
                       --input new_patients.csv \
                       --output predictions.csv \
                       --threshold 0.5

Input CSV must contain the same feature columns used in training:
    gender, age, hypertension, heart_disease, ever_married, work_type,
    Residence_type, avg_glucose_level, bmi, smoking_status
(id and stroke columns, if present, are dropped automatically.)
"""

import argparse

import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Score new patients with a trained stroke-risk model.")
    parser.add_argument("--model", type=str, required=True, help="Path to best_model.joblib")
    parser.add_argument("--input", type=str, required=True, help="CSV of new patient records")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Where to write predictions")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for flagging high risk. Lower this (e.g. 0.3) "
             "to prioritize catching more true positives at the cost of more false alarms.",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = joblib.load(args.model)

    print(f"Loading input data from {args.input}...")
    df = pd.read_csv(args.input)

    df_features = df.copy()
    for col in ["id", "stroke"]:
        if col in df_features.columns:
            df_features = df_features.drop(columns=[col])

    print(f"Scoring {len(df_features)} records...")
    probabilities = model.predict_proba(df_features)[:, 1]

    result = df.copy()
    result["stroke_risk_probability"] = probabilities.round(4)
    result["stroke_risk_flag"] = np.where(probabilities >= args.threshold, "Yes", "No")

    result.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(f"Flagged {result['stroke_risk_flag'].eq('Yes').sum()} of {len(result)} records as high risk "
          f"(threshold={args.threshold}).")


if __name__ == "__main__":
    main()