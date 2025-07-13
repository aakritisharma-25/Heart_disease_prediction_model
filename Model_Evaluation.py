import streamlit as st

st.title("📈 Model Evaluation Metrics")
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
