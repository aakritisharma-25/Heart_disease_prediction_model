# CardioPredict
## ❤️ Heart Disease Prediction Model

A Machine Learning-based web app built with Streamlit that predicts the possibility of heart disease based on user input data like age, cholesterol level, chest pain type, etc.

---

🚀 **Live Demo**:  
[Click here to try the Heart Disease Prediction App](https://heartdiseasepredictionmodel-cmm6acmwh9h4zwg7rxygcj.streamlit.app/)


---

## 📌 Features

- User-friendly input form for medical data
- Machine Learning model predicts presence of heart disease
- Displays probability and diagnosis
- Visual output: Confusion Matrix and other analytics
- Fully deployed using Streamlit Cloud

---

## 🧠 ML Model

The model was trained using scikit-learn's algorithms on a standard heart disease dataset. It was serialized using `joblib` / `pickle` and integrated into a Streamlit frontend.

---

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🛠 How to Run Locally

```bash
git clone https://github.com/your-username/heart_disease_prediction_model.git
cd heart_disease_prediction_model
pip install -r requirements.txt
streamlit run app.py
