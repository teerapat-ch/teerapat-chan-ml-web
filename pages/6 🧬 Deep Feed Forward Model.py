import streamlit as st
import numpy as np
import torch
import torch.nn as nn

class DeepFFNN(nn.Module):
    def __init__(self):
        super(DeepFFNN, self).__init__()
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.batch_norm(x)
        x = self.sigmoid(self.fc3(x))
        return x

model = DeepFFNN()
model.load_state_dict(torch.load("models/deep_ff_model.pth"))
model.eval()

st.header("Deep Feed Forward Model", divider="grey")
st.header("Heart Disease Prediction")
st.write("(อย่าเปลี่ยนค่าเร็วไปนะครับ 🙏)")

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
thallium = st.selectbox("Thallium", (3, 6, 7))
fbs = 1 if st.checkbox("FBS Over 120") else 0
exercise = 1 if st.checkbox("Exercise Angina") else 0

#####################################################

submit_button = st.button("Submit")

if submit_button:
    input_data = np.array([[age, mapped_gender, chest_pain, bp, cholesterol, fbs, ekg, hr, exercise, st_de, slope_st, vessel, thallium]], dtype=np.float32)
    input_tensor = torch.tensor(input_data)

    with torch.no_grad():
        prediction = model(input_tensor)

    st.subheader("Prediction Result")
    probability = prediction.item()
    st.write(f"Prediction Probability: {probability:.2f}")
    if probability > 0.5:
        st.success("The model predicts **Heart Disease**")
    else:
        st.success("The model predicts **No Heart Disease**")
