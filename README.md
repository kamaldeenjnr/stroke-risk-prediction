
Predicting Stroke Risk from Patient Health Data

A machine learning case study in building for a real-world constraint: minimizing missed diagnoses on a severely imbalanced clinical dataset.

Dataset: Available upon request

__https://github.com/kamaldeenjnr/stroke-risk-prediction__

**The Problem**

Stroke is one of the leading causes of death and long-term disability worldwide, and early risk identification is one of the few levers that actually changes outcomes. This project builds a machine learning pipeline that estimates a patient's stroke risk from routinely collected health data age, hypertension, heart disease history, average glucose level, BMI, smoking status, and related demographic and lifestyle attributes.

The dataset itself presented the project's central challenge: only **4.9% of patients** in the data had experienced a stroke. A model trained naively on this data could report 95%+ accuracy while never correctly identifying a single at-risk patient a textbook failure mode in medical screening applications, and the exact failure mode this project was built to avoid.
The Core Design Decision

In a clinical screening context, the cost of a **false negative** (telling an at-risk patient they're fine) is far higher than the cost of a **false positive** (flagging a healthy patient for a closer look). A missed stroke risk can cost a life; an unnecessary follow-up test costs time and money.

This project makes that tradeoff explicit rather than leaving it as an accident of the metric. Instead of optimizing for raw accuracy, the model selection and threshold were tuned to **maximize recall** (the share of actual stroke cases the model correctly catches) while keeping the resulting false-positive rate at a workable level for a screening tool not a diagnostic one.
That distinction screening tool vs. diagnostic tool shaped every decision below.
Approach

**1. Handling class imbalance**

With a 4.9% positive rate, standard training would bias the model heavily toward predicting "no stroke" for everyone. **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training data to synthetically balance the classes, giving the model enough signal on the minority class to learn from.

**2. Model comparison**
Three candidate models were trained and evaluated under cross-validation:
- Logistic Regression
- Random Forest
- XGBoost
Each was tuned via cross-validated hyperparameter search, optimizing for ROC-AUC to get a fair, threshold-independent comparison before selecting a final model and decision threshold.
**3. Model selection**
**Logistic Regression**- not the more complex ensemble methods came out on top. This is worth stating plainly rather than glossing over: on this dataset, the simpler, more interpretable model generalized better than Random Forest or XGBoost. In a healthcare context, that's a genuine advantage, not just a tiebreaker a linear model's coefficients are directly explainable to a clinician in a way a boosted tree ensembles aren't.
**4. Threshold tuning**
Rather than using the default 0.5 classification threshold, the decision threshold was tuned specifically to favor recall, consistent with the screening-tool framing above.
## Results
| Metric | Score |
|---|---|
| **Recall (Sensitivity)** | **80%** |
| **ROC-AUC** | **0.841** |
| Accuracy | 74% |
| Precision | 13.5% |

**What this means in practice:** out of every 50 patients who will actually go on to have a stroke, the model correctly flags 40 of them. It does raise a meaningful number of false alarms along the way roughly 6 flagged patients for every 1 true case which is the direct, intentional cost of prioritizing recall. In a screening context, that's a defensible trade: a false alarm prompts a follow-up conversation; a missed case doesn't get a second chance.
The 0.841 ROC-AUC indicates strong overall discriminative ability between the two classes across all thresholds, independent of the specific operating point chosen.
## What's in the Repository
```
stroke-risk-prediction/
├── train_model.py          # Full training pipeline: preprocessing, SMOTE, model comparison, tuning
├── predict.py               # Inference script for scoring new patients
├── requirements.txt
├── data/                    # Source dataset
├── models/
│   └── best_model.joblib    # Trained, serialized Logistic Regression model
├── outputs/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── glucose_vs_stroke.png
│   └── stroke_rate_by_age_group.png
└── new_patients_sample.csv  # Example input for predict.py
```
The trained model is included so anyone cloning the repo can run `predict.py` against new patient data immediately, without retraining from scratch.
## Tech Stack
`Python` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` · `pandas` · `matplotlib`

## Reflection

The instinct in most first ML projects is to chase the highest accuracy number and stop there. The more useful skill — and the one this project was built to practice — is recognizing when accuracy is the *wrong* metric for the problem in front of you, choosing the right one deliberately, and being able to explain that choice and its consequences clearly. A model that's 74% accurate but catches 80% of real stroke cases is a better screening tool than one that's 95% accurate and catches almost none — and being able to say precisely why is the actual deliverable here, alongside the code.

Full code, training pipeline, and outputs available in the https://github.com/kamaldeenjnr/stroke-risk-prediction
