import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib

# โหลดโมเดลและ scaler
model = tf.keras.models.load_model("models/deep_ff_model.keras")
scaler = joblib.load("models/scaler.pkl")

st.header("Deep Feed Forward Model", divider="grey")
st.header("Heart Disease Prediction")

################### SESSION STATE SETUP ###################
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None

################### INPUT SECTION ###################

age = st.number_input("Age", min_value=0, max_value=100, value=25)

gender_options = {"Male": 1, "Female": 0}
gender = st.selectbox("Gender", list(gender_options.keys()))
mapped_gender = gender_options[gender]

chest_pain = st.selectbox("Chest Pain Type", (1, 2, 3, 4))

bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)

ekg = st.selectbox("EKG Result", (0, 1, 2))

hr = st.number_input("Max HR", min_value=70, max_value=300, value=150)
st_de = st.number_input("ST Depression", min_value=0.0, max_value=7.0, step=0.1, value=1.0)

slope_st = st.selectbox("Slope of ST", (1, 2, 3))

vessel = st.selectbox("Number of Vessels Fluro", (0, 1, 2, 3))

thallium = st.selectbox("Thallium", (3, 7))

# แปลงค่าจาก checkbox เป็น 0/1
fbs = 1 if st.checkbox("FBS Over 120") else 0
exercise = 1 if st.checkbox("Exercise Angina") else 0

#####################################################

submit_button = st.button("Submit")

if submit_button:
    st.session_state.input_data = np.array([[age, mapped_gender, chest_pain, bp, cholesterol, fbs, ekg, hr, exercise, st_de, slope_st, vessel, thallium]], dtype=float)

    if st.session_state.input_data.shape[1] != scaler.n_features_in_:
        st.error(f"Input features ไม่ตรงกับ Scaler! ต้องการ {scaler.n_features_in_} ฟีเจอร์ แต่ได้รับ {st.session_state.input_data.shape[1]} ฟีเจอร์")
    else:
        scaled_input = scaler.transform(st.session_state.input_data)
        st.session_state.prediction = model.predict(scaled_input)

        st.subheader("Prediction Result")
        if st.session_state.prediction.shape[1] > 1:  # Multi-class
            predicted_class = np.argmax(st.session_state.prediction)
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {max(st.session_state.prediction[0]) * 100:.2f}%")
        else:  # Binary classification
            probability = st.session_state.prediction[0][0]
            st.write(f"Prediction Probability: {probability:.2f}")
            if probability > 0.5:
                st.success("The model predicts **Heart Disease**")
            else:
                st.success("The model predicts **No Heart Disease**")
