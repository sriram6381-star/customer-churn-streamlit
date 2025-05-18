import streamlit as st
import pandas as pd
import pickle

# Load model and preprocessor
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

st.title("Customer Churn Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload customer data CSV", type=['csv'])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove extra spaces

    st.subheader("Raw Data")
    st.write(df.head())

    # Preprocess and predict
    X = preprocessor.transform(df)
    predictions = model.predict(X)
    df['Churn_Predicted'] = predictions

    st.subheader("Predictions")
    st.write(df[['Churn_Predicted']])