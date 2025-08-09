import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("BOD Soft Sensor — Minimal Demo")

st.write("Load a trained model (ExtraTrees or MLP) and a CSV of features to get predictions.")

model_file = st.file_uploader("Model (.joblib)", type=["joblib"])
data_file = st.file_uploader("Data (.csv)", type=["csv"])

if model_file and data_file:
    model_path = Path("uploaded_model.joblib")
    model_path.write_bytes(model_file.getvalue())
    pipe = joblib.load(model_path)

    df = pd.read_csv(data_file)
    target = st.text_input("Target column to ignore (if present)", "F_BOD")
    if target in df.columns:
        df = df.drop(columns=[target])

    if st.button("Predict"):
        preds = pipe.predict(df)
        st.write("First 20 predictions:")
        st.dataframe(pd.DataFrame({"prediction": preds}).head(20))