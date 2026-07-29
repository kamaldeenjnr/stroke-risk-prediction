# Stroke Risk Prediction — Real ML Project (Healthcare)

Predicts a patient's risk of having a stroke based on demographic and health
attributes, using the public **Stroke Prediction Dataset**.

> **Disclaimer:** This is a portfolio/educational project, not a medical device
> and not clinically validated. It's built on a small public dataset, it finds
> correlations (not causal mechanisms), and it must never be used for actual
> diagnosis or treatment decisions. Frame it in your portfolio as a
> risk-flagging/triage-support proof of concept, not a diagnostic tool.

This is a full pipeline, not a toy notebook:
- Real data cleaning (missing BMI values, encoding categorical health/lifestyle features)
- **Severe class imbalance handling** — only ~5% of patients in this dataset had
  a stroke, which is the actual hard part of this problem and the part most
  toy tutorials skip
- Model comparison (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning (RandomizedSearchCV, 5-fold CV, ROC-AUC scoring)
- Evaluation that reports **recall** prominently, not just accuracy — in a
  screening problem like this, missing an actual stroke case (false negative)
  is far more costly than a false alarm, and a model can hit 95% accuracy
  by just predicting "no stroke" for everyone. This pipeline calls that out.
- Saved, reloadable model (`.joblib`) + a `predict.py` for scoring new patients

---

## 1. Get the data (do this first)

**Kaggle — "Stroke Prediction Dataset" by fedesoriano**
1. Create a free account at https://www.kaggle.com
2. Go to: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
3. Click **Download** → you'll get `healthcare-dataset-stroke-data.csv`
4. Place it in this project's `data/` folder (keep the filename as-is).

**Via Kaggle API (alternative)**
```bash
pip install kaggle
# API token from https://www.kaggle.com/settings -> "Create New Token" -> ~/.kaggle/kaggle.json
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p data/ --unzip
```

You should end up with:
```
stroke_prediction/data/healthcare-dataset-stroke-data.csv
```

Columns you'll find in it: `id, gender, age, hypertension, heart_disease,
ever_married, work_type, Residence_type, avg_glucose_level, bmi,
smoking_status, stroke` (target: 1 = had a stroke, 0 = did not).

Note: `bmi` has some missing values encoded as `"N/A"` strings — the script
handles this (imputes with the median) rather than silently dropping rows.

---

## 2. Set up the environment

```bash
cd stroke_prediction
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the model

```bash
python train_model.py --data data/healthcare-dataset-stroke-data.csv
```

This will:
1. Clean the data (fix missing BMI, drop the `id` column, encode target)
2. Split into train/test (80/20, stratified on stroke outcome — critical
   given how rare positive cases are)
3. Balance the **training** data only with SMOTE (test set stays untouched
   and realistic, so evaluation numbers reflect real-world class balance)
4. Train + tune Logistic Regression, Random Forest, and XGBoost
5. Select the best model by cross-validated ROC-AUC
6. Evaluate on the held-out test set
7. Save everything

### What you'll get afterward
```
models/
  best_model.joblib
outputs/
  metrics.json                    <- accuracy, precision, recall, F1, ROC-AUC
  classification_report.txt
  confusion_matrix.png
  roc_curve.png
  feature_importance.png
  stroke_rate_by_age_group.png    <- EDA insight
  glucose_vs_stroke.png           <- EDA insight
```

## 4. Score new patients with the trained model

```bash
python predict.py --model models/best_model.joblib --input new_patients.csv --output predictions.csv
```

`new_patients.csv` needs the same columns as training data minus `stroke`
(and minus `id`). Output adds `stroke_risk_probability` and
`stroke_risk_flag` (Yes/No at a 0.5 threshold — worth discussing in your
write-up whether that threshold is even the right one for a screening use
case, since you may want higher recall at the cost of more false positives).

---

## Why this is a strong portfolio piece

- **Real imbalance problem.** Most churn/HR tutorials use datasets that are
  20-30% positive class. This one is ~5% — you have to actually justify
  your technique choice (SMOTE vs. class weights vs. threshold tuning)
  instead of getting lucky with a default classifier.
- **The right metric isn't accuracy.** This project forces you to explain
  *why* you're optimizing for recall/ROC-AUC over accuracy in a screening
  context — a good, defensible talking point in an interview.
- **Deployable artifact**, not just a notebook — `predict.py` is real
  inference code you could wrap in a Flask/FastAPI endpoint for a live demo.

## Suggested portfolio write-up angle

*"Built a stroke risk screening model on an imbalanced clinical dataset
(~5% positive rate). Optimized for recall rather than accuracy, since in a
screening context a missed positive is far costlier than a false alarm.
Achieved X% recall / Y ROC-AUC on held-out data, and discussed the
precision/recall tradeoff and threshold selection as a product decision,
not just a modeling one."*
