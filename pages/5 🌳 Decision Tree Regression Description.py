import streamlit as st
import pandas as pd
import numpy as np

data = pd.read_csv("datasets/salary_dataset.csv")
df = pd.DataFrame({'Years': data['Years_of_Experience'], 'Salary': data['Salary']})
df = df.dropna()

st.header("Decision Tree Regression Description", divider="grey")

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
st.write("Decision Tree Regression เป็นอัลกอริธึมที่ใช้โครงสร้างต้นไม้ในการแบ่งข้อมูลออกเป็นกลุ่มย่อย แล้วใช้ค่าเฉลี่ยของแต่ละกลุ่มเป็นค่าพยากรณ์ของโมเดล")
st.write("เริ่มจากการแบ่งข้อมูลออกเป็นกลุ่ม โดยที่ข้อมูลภายในกลุ่มจะต้องมีความแปรปรวนน้อยที่สุด")
st.write("หลังจากที่ได้กลุ่มออกมาแล้วก็จะทำการสร้างต้นไม้แยกออกมาจนครบทุกกลุ่ม")
st.write("ในการ predict ค่าที่รับมา เมื่อจัดเข้ากลุ่มใดกลุ่มหนึ่งแล้วจะให้ค่า output ที่ predict ได้จะเป็นค่าเฉลี่ยของข้อมูลทั้งหมดในกลุ่ม")

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

dtree = Ridge(random_state=0)
dtree.fit(years_list, salary_list)
'''
st.code(train_code, language="python")

st.write("Export ตัวโมเดลที่ train แล้วออกมา")
export_code = '''
port joblib
joblib.dump(dtree, 'dtree_model.pkl')
'''
st.code(export_code, language="python")

st.subheader("2. Deploy model")
st.write("Import ตัวโมเดลเข้ามายัง streamlit")
import_code = '''
import joblib
model = joblib.load("models/dtree_model.pkl")
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