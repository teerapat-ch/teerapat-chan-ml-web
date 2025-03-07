import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv("datasets/heart_disease.csv")
data_cleaned = data.dropna()

st.header("Deep Feed Forward Description", divider="grey")

##### การเตรียมข้อมูล #####
st.subheader("การเตรียมข้อมูล")
st.write("ข้อมูลที่ใช้ train คือข้อมูลการเป็นโรคหัวใจเทียบกับอายุ เพศ และข้อมูลสุขภาพอื่นๆ ตัวของ dataset ดาวน์โหลดจากเว็ปไซต์ kaggle.com")
st.link_button("แหล่งที่มาข้อมูล: Kaggle.com", "https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction")
st.write("ตัวอย่างข้อมูล")
st.write(data.head(10))
st.write("ข้อมูลในบาง row มี field ที่ไม่มีข้อมูล dataset จึงมีความไม่สมบูรณ์ เนื่องจาก row ที่ไม่สมบูรณ์มีไม่มากจึงทำการลบออกจาก dataset ด้วยคำสั่ง")
st.code("df = df.dropna()", language="python")
st.write("ซึ่งจะทำการกรองให้เหลือแค่ row ที่มีข้อมูลครบ")

##### ทฤษฎีของอัลกอริทึม #####
st.subheader("ทฤษฎีของอัลกอริทึม")
st.write("Deep Feed Forward Neural Network เป็นโมเดล Feed Forward ที่มี hidden layer มากกว่า 1")
st.write("ใน input layer จะทำการรับค่าที่ใช้ในการ predict ใน hidden layers จะทำการคำนวณแบบเชิงเส้นด้วยฟังก์ชั่น activation เช่น ReLU, Sigmoid และใน output layer จะให้ค่า result ออกมา")

##### ขั้นตอนการพัฒนา #####
st.subheader("ขั้นตอนการพัฒนา")
st.subheader("1. สร้าง model")

st.write("ทำการ import dataset เข้ามา")
st.code('''
data = pd.read_csv("/content/heart_disease.csv")
''', language="python")

st.write("ทำการ map ผลลัพธ์ให้เป็น 1 และ 0 + clean ข้อมูลที่ข้อมูลไม่สมบูรณ์")
map = st.code('''
df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
df = df.dropna()
''')

st.write("แบ่งข้อมูลเป็น x และ y")
st.code('''
y = df['Heart Disease']
x = df.drop(columns = ['Heart Disease'])
''')

st.write("แบ่งข้อมูลไว้ train และ test")
st.code('''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
''')

st.write("สร้างโมเดล Deep Feed Forward")
st.code('''
import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
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

model = FeedForwardNN()

print(model)
''')

st.write("ตั้งค่าให้ optimizer")
st.code('''
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
''')

st.write("แปลงข้อมูลที่จะ train เข้าไปยัง model")
st.code('''
from torch.utils.data import DataLoader, TensorDataset

x_tensor = torch.tensor(x.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(x_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
''')

st.write("train ข้อมูลตามจำนวน epochs")
st.code('''
for epoch in range(20):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
''')

st.write("ทดสอบความแม่นยำของโมเดล")
st.code('''
model.eval()
with torch.no_grad():
    x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    y_pred = model(x_test_tensor)
    y_pred = (y_pred > 0.5).float()

    accuracy = (y_pred == y_test_tensor).float().mean()

print(f'Test Accuracy: {accuracy.item():.4f}')
''')

st.write("Export โมเดล")
st.code('''
torch.save(model.state_dict(), 'deep_ff_model.pth')
''')

st.subheader("2. Deploy model")

st.write("Import ตัวโมเดลเข้ามายัง streamlit")
st.code('''
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
''')

st.write("รับ input จากผู้ใช้")
st.code('''
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
''')

st.write("ทำการ predict")
st.code('''
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
''')