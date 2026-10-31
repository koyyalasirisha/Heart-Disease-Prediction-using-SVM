"""Streamlit dashboard for educational heart-disease risk classification."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.prediction import predict_patient
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Heart Disease Prediction using SVM",
    page_icon="♥",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #f5f7f6; }
    [data-testid="stSidebar"] { background: #e8efeb; }
    .hero { padding: 1.3rem 1.6rem; border-left: 5px solid #14866d; background: #ffffff; }
    .notice { padding: .8rem 1rem; border-radius: 4px; background: #fff3d6; color: #614700; }
    div[data-testid="stMetric"] { background: #fff; padding: .8rem; border: 1px solid #dce4df; border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Heart Health · SVM")
page = st.sidebar.radio(
    "Navigate",
    ["Home / Dashboard", "Patient Prediction", "Model Performance", "Dataset Exploration"],
)
st.sidebar.caption("Educational machine-learning demonstration")


def read_metrics() -> dict | None:
    metrics_file = MODEL_DIR / "metrics.json"
    if not metrics_file.is_file():
        return None
    try:
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def show_dataset_missing() -> None:
    st.error(f"Dataset not found at `{DATA_PATH}`. Add a compatible CSV file to the `data` folder.")
    st.markdown("Expected columns: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`.")


def metric_tiles(metrics: dict) -> None:
    columns = st.columns(4)
    for column, label, key in zip(
        columns,
        ["Accuracy", "Precision", "Recall", "F1-score"],
        ["accuracy", "precision", "recall", "f1_score"],
    ):
        column.metric(label, f"{metrics[key]:.1%}")


if page == "Home / Dashboard":
    st.markdown(
        "<div class='hero'><h1>Heart Disease Prediction using SVM</h1>"
        "<p>A supervised-learning demonstration that classifies patient records with a Support Vector Machine.</p></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Objective")
    st.write(
        "Explore the Cleveland heart-disease dataset, review a held-out model evaluation, "
        "and submit a patient feature profile for an educational classification."
    )
    metrics = read_metrics()
    try:
        dataset = load_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        dataset = None
        st.info(str(error))

    if dataset is not None:
        first, second, third = st.columns(3)
        first.metric("Records", f"{len(dataset):,}")
        second.metric("Input features", len(FEATURE_COLUMNS))
        third.metric("Disease prevalence", f"{dataset[TARGET_COLUMN].mean():.1%}")
    if metrics:
        st.subheader("Held-out model snapshot")
        metric_tiles(metrics)
    else:
        st.warning("Model artifacts are not trained yet. Follow the README training command to create them.")

    st.subheader("Project at a glance")
    left, right = st.columns(2)
    with left:
        st.markdown("**Technologies**")
        st.write("Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Streamlit · Joblib")
        st.markdown("**Classifier**")
        st.write("Support Vector Classifier (SVC), with a configurable linear, RBF, or polynomial kernel.")
    with right:
        st.markdown("**Dataset**")
        st.write("UCI Heart Disease, Cleveland subset: 13 clinical attributes and a disease-presence target.")
        st.markdown("**Interpretation**")
        st.markdown("<div class='notice'>This prediction is for educational purposes only and is not a medical diagnosis.</div>", unsafe_allow_html=True)

elif page == "Patient Prediction":
    st.title("Patient Prediction")
    st.caption("Enter the 13 features expected by the trained Cleveland dataset model.")
    st.markdown("<div class='notice'>This prediction is for educational purposes only and is not a medical diagnosis.</div>", unsafe_allow_html=True)
    st.subheader("Patient Information")
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
            sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1], index=1)
            cp = st.selectbox("Chest pain type (1-4)", [1, 2, 3, 4], index=0)
            trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=300, value=130)
            chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=800, value=240)
        with col2:
            fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1], index=0)
            restecg = st.selectbox("Resting ECG result (0-2)", [0, 1, 2], index=0)
            thalach = st.number_input("Maximum heart rate achieved", min_value=40, max_value=250, value=150)
            exang = st.selectbox("Exercise-induced angina", [0, 1], index=0)
        with col3:
            oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox("ST segment slope (1-3)", [1, 2, 3], index=1)
            ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3], index=0)
            thal = st.selectbox("Thalassemia code (3, 6, or 7)", [3, 6, 7], index=0)
        submitted = st.form_submit_button("Predict Heart Disease", type="primary", width="stretch")

    if submitted:
        patient = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        try:
            prediction = predict_patient(patient, MODEL_DIR)
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error("Heart Disease Detected")
            else:
                st.success("No Heart Disease Detected")
            st.caption("Model output only; not a diagnosis or substitute for professional medical advice.")
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))

elif page == "Model Performance":
    st.title("Model Performance")
    metrics = read_metrics()
    if not metrics:
        st.warning("No evaluation is available yet. Train the model first.")
    else:
        st.caption(f"Kernel: {metrics['kernel']} · Held-out records: {metrics['test_records']} · Split seed: {metrics['random_state']}")
        metric_tiles(metrics)
        st.subheader("Confusion Matrix")
        matrix = metrics["confusion_matrix"]
        labels = [[f"TN\n{matrix[0][0]}", f"FP\n{matrix[0][1]}"], [f"FN\n{matrix[1][0]}", f"TP\n{matrix[1][1]}"]]
        figure, axis = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            cmap="crest",
            cbar=False,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=axis,
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        st.pyplot(figure, width="content")
        plt.close(figure)
        st.subheader("Classification Report")
        st.code(metrics["classification_report"], language="text")
        st.caption("TN = correctly predicted no disease; FP = false alarm; FN = missed disease; TP = correctly predicted disease.")

elif page == "Dataset Exploration":
    st.title("Dataset Exploration")
    try:
        dataset = load_dataset(DATA_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        show_dataset_missing()
        st.caption(str(error))
    else:
        st.subheader("Dataset Analysis")
        no_disease, disease = st.columns(2)
        no_disease.metric("Records", f"{len(dataset):,}")
        disease.metric("Features", len(FEATURE_COLUMNS))
        missing = dataset[FEATURE_COLUMNS].isna().sum()
        st.markdown("**Missing values by feature**")
        st.dataframe(missing.rename("Missing values"), width="stretch")
        st.markdown("**Class distribution**")
        class_counts = dataset[TARGET_COLUMN].map({0: "No Heart Disease", 1: "Heart Disease"}).value_counts()
        figure, axis = plt.subplots(figsize=(6, 3.5))
        sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, legend=False, palette=["#14866d", "#db6b52"], ax=axis)
        axis.set_ylabel("Records")
        axis.set_xlabel("")
        st.pyplot(figure, width="content")
        plt.close(figure)

        plot_columns = st.columns(3)
        for column, feature, title, color in zip(
            plot_columns,
            ["age", "chol", "thalach"],
            ["Age distribution", "Cholesterol distribution", "Maximum heart rate"],
            ["#14866d", "#db6b52", "#3c78a8"],
        ):
            with column:
                st.markdown(f"**{title}**")
                figure, axis = plt.subplots(figsize=(4, 3))
                sns.histplot(dataset[feature].dropna(), kde=True, color=color, ax=axis)
                axis.set_xlabel(feature)
                st.pyplot(figure, width="stretch")
                plt.close(figure)

        st.markdown("**Feature correlation**")
        figure, axis = plt.subplots(figsize=(11, 8))
        sns.heatmap(dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(), cmap="vlag", center=0, annot=True, fmt=".1f", ax=axis)
        st.pyplot(figure, width="stretch")
        plt.close(figure)

        st.markdown("**Basic statistics**")
        st.dataframe(dataset[FEATURE_COLUMNS].describe().T, width="stretch")
        with st.expander("Preview cleaned records"):
            st.dataframe(dataset.head(10), width="stretch")
