import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv("datasets/salary_dataset.csv")
df = pd.DataFrame({'Years': data['Years_of_Experience'], 'Salary': data['Salary']})
df = df.dropna()

st.header("Ridge Description", divider="grey")

# การเตรียมข้อมูล
st.subheader("การเตรียมข้อมูล")
st.write("ข้อมูลที่ใช้ในการ train คือข้อมูลเงินเดือนเทียบกับปีการทำงาน โดยทำการสร้าง dataset จาก ChatGPT")
st.write(data.head(10))
st.write("ตัวอย่างข้อมูลของ dataset")
st.write("ข้อมูลในบาง row มี field ที่ไม่มีข้อมูล dataset จึงมีความไม่สมบูรณ์ เนื่องจาก row ที่ไม่สมบูรณ์มีไม่มากจึงทำการลบออกจาก dataset ด้วยคำสั่ง")

st.code("df = df.dropna()", language="python")

st.write("ซึ่งจะทำการกรองให้เหลือแค่ row ที่มีข้อมูลครบ")
st.write(df.head(10))
st.write("ตัวอย่างข้อมูลของ dataset หลังจากทำการ clean แล้ว")

# ทฤษฎีของอัลกอริทึม
st.subheader("ทฤษฎีของอัลกอริทึม")
st.write("Ridge Regression เป็นหนึ่งในเทคนิค Regularization ที่ใช้กับ Linear Regression เพื่อลดปัญหา overfitting โดยเพิ่ม penalty term เข้าไปในฟังก์ชัน loss เพื่อจำกัดขนาดของค่าสัมประสิทธิ์ (coefficients)")

st.latex("Loss Function = \sum_{i=1}^{n} (y-\hat{y})^2 + λ\sum_{j=1}^{p} β^2_j")

st.latex("y_i\quadค่าจริงของตัวแปรที่ต้องการทำนาย")
st.latex("\hat{y_i}\quadคือค่าที่ได้จากการทำนายของโมเดล")
st.latex("β_j\quadคือค่าสัมประสิทธิ์ของตัวแปร\quad j")
st.latex("n\quadคือจำนวนตัวอย่าง")
st.latex("p\quadคือจำนวน\quad features")
st.latex("λ\quadคือพารามิเตอร์ที่ควบคุมขนาดของ\quad penalty")

# ขั้นตอนการพัฒนา
st.subheader("ขั้นตอนการพัฒนา")
st.subheader("1. สร้าง model")

st.write("ทำการ import dataset เข้ามา และ clean ข้อมูล")
import_data_code = '''
data = pd.read_csv("/content/salary_dataset.csv")
df = pd.DataFrame({'Years': data['Years_of_Experience'], 'Salary': data['Salary']})
df = df.dropna()
'''
st.code(import_data_code, language="python")

st.write("นำข้อมูลไป train กับโมเดล ridge")
train_code = '''
years = np.array(df['Years']).reshape(-1, 1)
salary = np.array(df['Salary'])

years_list = list(years)
salary_list = list(salary)

ridge = Ridge(alpha=1)
ridge.fit(years_list, salary_list)
'''
st.code(train_code, language="python")

st.write("Export ตัวโมเดลที่ train แล้วออกมา")
export_code = '''
port joblib
joblib.dump(ridge, 'ridge_model.pkl')
'''
st.code(export_code, language="python")

st.subheader("2. Deploy model")
st.write("Import ตัวโมเดลเข้ามายัง streamlit")
import_code = '''
import joblib
model = joblib.load("models/ridge_model.pkl")
'''
st.code(import_code, language="python")

st.write("หลังจากที่ import โมเดลเข้ามาแล้ว สามารถรับค่าจากผู้ใช้มา predict ได้เลย")
predict_code = '''
input = st.number_input("Enter years of experience between 0-40", min_value=0, max_value=40)
prediction = model.predict(np.array([[input]]))
'''
st.code(predict_code, language="python")

st.write("แสดงผลลัพธ์การ predict ผ่านกราฟ")
result = '''
def generate_graph(input, prediction, years_list, salary_list):
    plt.figure(figsize=(10,6))
    plt.scatter(years_list, salary_list, color='blue', label="All data Points")
    plt.scatter(input, prediction[0], color='red', label="New data point")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Salary Prediction')
    plt.legend()
    st.pyplot(plt)
'''
st.code(result, language="python")