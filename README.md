# Heart Disease Prediction using Support Vector Machine (SVM)

## Abstract

This educational machine-learning project trains a Support Vector Classifier (SVC) to classify records from the Cleveland Heart Disease dataset as either no disease (0) or disease present (1). A Streamlit interface provides dataset exploration, held-out evaluation, and individual patient predictions. It is a demonstration of a classification workflow, not a clinical tool.

## Problem Statement

Heart-disease datasets contain a mixture of numeric measurements and encoded clinical categories. This project demonstrates how to prepare those features consistently, train a supervised classifier, and communicate its evaluation and limitations through an accessible local web application.

## Objectives

- Load and inspect the Cleveland heart-disease dataset.
- Handle missing values without leaking test-set information into training.
- Train an SVM classifier with a reproducible stratified holdout split.
- Report accuracy, precision, recall, F1-score, a labeled confusion matrix, and a classification report.
- Provide dataset visualizations and a form for individual educational predictions.

## Technologies

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, and Joblib.

## Dataset

The preferred source is the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), using the Cleveland subset. Place its CSV at `data/heart.csv`. The project accepts the usual named Cleveland fields and the original headerless 14-column Cleveland data format. The original `num` target is converted to binary: 0 means no disease and 1-4 mean disease present.

Required features are `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, and `thal`. The target can be named `target`, `num`, `condition`, or another recognized common alias. See [data/README.md](data/README.md) for details.

## Features

- Dataset dimensions, preview, missing values, duplicates, class distribution, and summary statistics.
- Age, cholesterol, and maximum-heart-rate distributions plus a feature correlation heatmap.
- Configurable `linear`, `rbf`, or `poly` SVC kernel; `linear` is the default for an interpretable baseline and a straightforward demonstration.
- Saved model, median imputer, and scaler artifacts, plus held-out metrics.
- Streamlit pages for dashboard, patient prediction, model performance, and dataset exploration.

## Methodology

1. Normalize common column names and numeric values; convert `?` to missing values.
2. Drop rows without a target and exact duplicate records. Convert the target to binary using `target > 0`.
3. Split data into 80% training and 20% testing with stratification and `random_state=42`.
4. Fit a median `SimpleImputer` on training features only. Apply it to the test features.
5. Fit `StandardScaler` on imputed training features only. Transform the test set with that same scaler.
6. Fit an SVC and evaluate predictions on the held-out test set.

The Cleveland categorical features are already represented as integer category codes. They are kept in that form and standardized with the numeric measurements. This keeps the feature contract simple for the demonstration; it does not imply that category codes have a true continuous interpretation.

## SVM Algorithm

A Support Vector Machine selects a decision boundary that maximizes the margin between classes. `SVC(kernel="linear")` is the default model. The `rbf` kernel can represent nonlinear boundaries, while `poly` adds polynomial feature interactions. Kernel comparisons should use the same fixed train/test split or cross-validation; test-set performance should not be used to tune the final model.

## Evaluation Metrics

- **Accuracy:** fraction of all held-out records classified correctly (also shown as a percentage).
- **Precision:** fraction of predicted disease cases that are positive in the dataset.
- **Recall:** fraction of disease cases in the dataset found by the model.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** TN and TP are correct predictions; FP and FN are the two error types. The app labels all four cells.
- **Classification report:** per-class precision, recall, F1-score, and support.

These metrics describe this dataset split only and do not establish clinical performance.

## Project Structure

```text
heart-disease-svm/
├── data/
│   ├── README.md
│   └── heart.csv                 # Add the dataset here
├── models/                       # Created by training; artifacts are git-ignored
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── prediction.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

From the `heart-disease-svm` directory, run in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the commands with `.venv\Scripts\python.exe` instead of `python`.

## Train and Evaluate

Add the dataset first, then train the default linear model:

```powershell
python -m src.train_model --data data/heart.csv --kernel linear
```

Choose `linear`, `rbf`, or `poly` with `--kernel`. Training creates `models/svm_model.pkl`, `models/scaler.pkl`, `models/imputer.pkl`, and `models/metrics.json`. The preprocessing artifacts are fit on training rows only.

Recreate the held-out evaluation and confusion-matrix image with:

```powershell
python -m src.evaluate_model --data data/heart.csv
```

## Run the Application

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. Train the model before opening Patient Prediction or Model Performance. Dataset exploration can be used once `data/heart.csv` is present.

## Sample Prediction

The app form requests all 13 features in the same order as the training dataset. For programmatic use, call `src.prediction.predict_patient` with a dictionary keyed by the feature names. The returned value is 0 or 1, produced by the saved SVM; no prediction result is hard-coded. A sample record is not included because predictions depend on the specific dataset and its learned model.

## Error Handling

The loader reports missing files, missing required fields, empty datasets, and targets with only one class. Feature missing values are median-imputed from training data. Prediction checks required keys and finite numeric values, and reports missing model artifacts with the training command.

## Limitations

- The Cleveland dataset is small and may not represent current or diverse patient populations.
- Integer-coded categories are standardized as numeric features rather than one-hot encoded.
- A single train/test split gives a limited estimate of generalization.
- No probability calibration, external validation, fairness assessment, or clinical validation is provided.
- This project is for educational purposes only and must not be used for medical decisions.

## Future Enhancements

- Add stratified cross-validation and report variability across folds.
- Compare kernels using training-only cross-validation and tune hyperparameters.
- Add calibration, explainability tools, fairness checks, and external validation.
- Add robust schema and unit validation for alternate heart-disease CSV variants.
