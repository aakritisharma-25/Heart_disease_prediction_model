import streamlit as st
import pickle
import numpy as np

# Load model
import joblib

model = joblib.load("Heart_disease_model.pkl")


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Patient Input", "Model Evaluation", "Dataset Info"])

# Page 1: Patient Input
if page == "Patient Input":
    st.title("Heart Disease Prediction")
    st.markdown("### Enter Patient Details")

    # Patient details
    age = st.number_input("Patient Age", min_value=0, max_value=120)
    sex = st.selectbox("Patient Gender", ["male", "female"])

    st.markdown("### Medical Information")

    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
        help="Select the type of chest pain experienced."
    )

    trestbps = st.slider(
        "trestbps: Resting Blood Pressure (mm Hg)",
        0, 200, help="Resting blood pressure on admission."
    )

    chol = st.slider(
        "chol: Serum Cholesterol (mg/dl)",
        0, 600, help="Measured in mg/dl. Normal range: 125–200."
    )

    fbs = st.selectbox(
        "fbs: Fasting Blood Sugar > 120 mg/dl", ["True", "False"],
        help="Indicates whether fasting blood sugar is greater than 120."
    )

    restecg = st.selectbox(
        "restecg: Resting ECG Results",
        ["normal", "stt abnormality", "lv hypertrophy"],
        help="ECG results: normal, ST-T wave abnormality, or LV hypertrophy."
    )

    thalch = st.number_input(
        "thalch: Maximum Heart Rate Achieved", min_value=0, max_value=250
    )

    exang = st.selectbox(
        "exang: Exercise Induced Angina", ["True", "False"],
        help="Chest pain during physical exertion."
    )

    oldpeak = st.number_input(
        "oldpeak: ST Depression",
        help="ST depression induced by exercise relative to rest."
    )

    slope = st.selectbox(
        "slope: Slope of the Peak Exercise ST Segment",
        ["upsloping", "flat", "downsloping"]
    )

    ca = st.selectbox(
        "ca: Number of Major Vessels Colored (0-3)", [0, 1, 2, 3],
        help="Detected by fluoroscopy."
    )

    thal = st.selectbox(
        "thal: Thalassemia Status", ["normal", "fixed defect", "reversible defect"]
    )

    # Map categorical inputs to numerical values
    sex = 1 if sex == "male" else 0
    cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
    cp = cp_map[cp]
    fbs = 1 if fbs == "True" else 0
    restecg_map = {"normal": 0, "stt abnormality": 1, "lv hypertrophy": 2}
    restecg = restecg_map[restecg]
    exang = 1 if exang == "True" else 0
    slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
    slope = slope_map[slope]
    thal_map = {"normal": 1, "fixed defect": 2, "reversible defect": 3}
    thal = thal_map[thal]

    # Create input array
    features = np.array([
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalch, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(features)[0]
        disease_stage = ["No Heart Disease", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
        st.success(f"Prediction: {disease_stage[prediction]}")

# Page 2: Model Evaluation
elif page == "Model Evaluation":
    st.title("Model Evaluation Metrics")
    st.image("confusion_matrix.png", caption="Confusion Matrix")

    st.image("precision.png", caption="Precision, Recall, F1 by Class")
    st.image("model_compare.png", caption="Train/Test Accuracy Table")

    st.markdown("### Classification Report")
    st.code("""
    precision    recall  f1-score   support

    0       0.89      0.88      0.88       200
    1       0.87      0.88      0.87       199
    2       0.98      0.98      0.98       200
    3       0.87      0.87      0.87       200
    4       0.88      0.88      0.88       199

    accuracy                           0.90       998
    macro avg       0.90      0.90      0.90       998
    weighted avg    0.90      0.90      0.90       998
    """)

# Page 3: Dataset Info
elif page == "Dataset Info":
    st.title("Dataset Details")
    st.markdown("[UCI Heart Disease Dataset (Kaggle.com)](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)")

    st.markdown("""
    - This model is trained on a synthetic dataset derived from the [UCI Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci).
    - Synthetic data was created to have 5000 rows.
    - Original dataset had 16 features. Two features were dropped after preprocessing.
    - Final dataset has 14 columns:
        - One target column `num`:
            - 0 = No heart disease
            - 1, 2, 3, 4 = Increasing severity/stages of heart disease
        - 13 input features collected from the patient (as seen in Patient Input page).

    Features used:
    - Age, Gender (Sex), Chest Pain Type (`cp`), Resting Blood Pressure (`trestbps`), Serum Cholesterol (`chol`), Fasting Blood Sugar (`fbs`), Resting ECG (`restecg`), Max Heart Rate (`thalach`), Exercise Induced Angina (`exang`), ST Depression (`oldpeak`), Slope (`slope`), Major Vessels Colored (`ca`), Thalassemia (`thal`)
    """)
    
    st.markdown("### About kaggle Dataset")
    st.markdown("""
    **Context**  
    This is a multivariate type of dataset which means providing or involving a variety of separate mathematical or statistical variables — multivariate numerical data analysis. It is composed of 14 attributes which are:

    - age  
    - sex  
    - chest pain type  
    - resting blood pressure  
    - serum cholesterol  
    - fasting blood sugar  
    - resting electrocardiographic results  
    - maximum heart rate achieved  
    - exercise-induced angina  
    - oldpeak — ST depression induced by exercise relative to rest  
    - the slope of the peak exercise ST segment  
    - number of major vessels  
    - Thalassemia  

    This database includes 76 attributes, but all published studies relate to the use of a subset of 14 of them. The Cleveland database is the only one used by ML researchers to date.

    One of the major tasks on this dataset is to predict — based on the given attributes of a patient — whether that particular person has heart disease or not. Another experimental task is to diagnose and discover various insights from this dataset, which could help in understanding the problem better.
    """)

    st.markdown("### Column Descriptions")
    st.markdown("""
    1. **id** – Unique id for each patient  
    2. **age** – Age of the patient in years  
    3. **origin** – Place of study  
    4. **sex** – Male/Female  
    5. **cp** – Chest pain type (*typical angina, atypical angina, non-anginal, asymptomatic*)  
    6. **trestbps** – Resting blood pressure (*in mm Hg on admission to the hospital*)  
    7. **chol** – Serum cholesterol in mg/dl  
    8. **fbs** – Fasting blood sugar > 120 mg/dl  
    9. **restecg** – Resting electrocardiographic results  
    -*Values: normal, ST-T wave abnormality, left ventricular hypertrophy*  
    10. **thalach** – Maximum heart rate achieved  
    11. **exang** – Exercise-induced angina (*True/False*)  
    12. **oldpeak** – ST depression induced by exercise relative to rest  
    13. **slope** – Slope of the peak exercise ST segment  
    14. **ca** – Number of major vessels (0-3) colored by fluoroscopy  
    15. **thal** – *normal, fixed defect, reversible defect*  
    16. **num** – The predicted attribute (target)
    """)

