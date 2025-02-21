import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/ridge_model.pkl")

data = pd.read_csv("datasets/salary_dataset.csv")
df = pd.DataFrame({'Years': data['Years_of_Experience'], 'Salary': data['Salary']})
df = df.dropna()

st.header("Ridge Model", divider="grey")
input = st.number_input("Enter year of experience between 0-40", min_value=0, max_value=40)

prediction = model.predict(np.array([[input]]))

st.write(f"Salary = {prediction}")
st.scatter_chart(df, x="Years", y="Salary")