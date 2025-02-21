import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

model = joblib.load("models/ridge_model.pkl")

@st.cache_data
def generate_graph():
    plt.figure(figsize=(10,6))
    plt.scatter(years_list, salary_list, color='blue', label="All data Points")
    plt.scatter(input, prediction[0], color='red', label="New data point")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Salary Prediction')
    plt.legend()
    st.pyplot(plt)

data = pd.read_csv("datasets/salary_dataset.csv")
df = pd.DataFrame({'Years': data['Years_of_Experience'], 'Salary': data['Salary']})
df = df.dropna()
years = np.array(df['Years']).reshape(-1, 1)
salary = np.array(df['Salary'])
years_list = list(years)
salary_list = list(salary)

st.header("Ridge Model", divider="grey")
input = st.number_input("Enter years of experience between 0-40", min_value=0, max_value=40)
submit_button = st.button("Submit")

if submit_button:
    prediction = model.predict(np.array([[input]]))
    generate_graph()
    st.write(f"Years of experience = {input}")
    st.write(f"Salary = {prediction[0]:.2f}")