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
from keras import layers, regularizers

model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(13,)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation="sigmoid")
])

model.summary()
''')

st.write("ตั้งค่าให้ output เป็น binary(0-1)")
st.code('''
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
''')

st.write("fit ข้อมูลที่จะ train เข้าไปยัง model")
st.code('''
history_model = model.fit(x, y, epochs=20, validation_split=0.2, batch_size=16)
''')

st.write("ทดสอบความแม่นยำของโมเดล")
st.code('''
model.evaluate(x_test, y_test)
''')

st.write("Import โมเดล")
st.code('''
model.save('deep_ff_model.keras')
''')

st.subheader("2. Deploy model")

st.write("Import ตัวโมเดลเข้ามายัง streamlit")
st.code('''
import tensorflow as tf
model = tf.keras.models.load_model("models/deep_ff_model.keras")
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
input_data = np.array([[age, mapped_gender, chest_pain, bp, cholesterol, fbs, ekg, hr, exercise, st_de, slope_st, vessel, thallium]], dtype=float)
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    probability = prediction[0][0]
    st.write(f"Prediction Probability: {probability:.2f}")
    if probability > 0.5:
        st.success("The model predicts **Heart Disease**")
    else:
        st.success("The model predicts **No Heart Disease**")
''')