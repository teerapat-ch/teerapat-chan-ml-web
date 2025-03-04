import streamlit as st
import pandas as pd
import numpy as np

st.header("Deep Feed Forward Model", divider="grey")
st.header("Heart Disease Prediction")

age = st.number_input("Age", min_value = 0, max_value = 100)
gender = st.selectbox("Gender", ("Male", "Female"))
chest_pain = st.selectbox("Chest Pain Type", (1, 2, 3, 4))
bp = st.number_input("Blood Pressure", min_value = 0, max_value = 200)
cholesterol = st.number_input("Cholesterol", min_value = 100, max_value = 600)
ekg = st.selectbox("EKG Result", (0, 1, 2))
hr = st.number_input("Max HR" , min_value = 70, max_value = 300)
st_de = st.number_input("ST Depression", min_value = 0, max_value = 7)
slope_st = st.selectbox("Slope of ST", (1, 2, 3))
vessel = st.selectbox("Number of Vessels Fluro", (0, 1, 2, 3))
thallium = st.selectbox("Thallium", (3, 7))
fbs = st.checkbox("FBS Over 120")
exercise = st.checkbox("Exercise Angina")

submit_button = st.button("Submit")