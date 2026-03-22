# import
import streamlit as st
import numpy as np
import joblib

# load model
data = joblib.load("heart_model.h5")
model = data["model"]
imputer = data["imputer"]
scaler = data["scaler"]
st.title("Dự đoán bệnh tim")
st.write("Nhập thông tin của bạn để dự đoán nguy cơ mắc bệnh tim.")

# input
age = st.number_input("Tuổi", 20, 100, 50)
sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
cp = st.selectbox("Loại đau ngực", ["Không đau ngực", "Đau ngực không điển hình", "Đau ngực điển hình", "Không đau ngực"])
trestbps = st.number_input("Huyết áp", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
thalach = st.number_input("Nhịp tim tối đa", 60, 220, 150)
oldpeak = st.number_input("Giảm ST", 0.0, 10.0, 1.0)

# encode
sex = 1 if sex == "Nam" else 0
cp_mapping = {
    "Không đau ngực": 0,
    "Đau ngực không điển hình": 1,
    "Đau ngực điển hình": 2,
    "Không đau ngực": 3
}
cp = cp_mapping[cp]

# input
input_data = np.array([[age, sex, cp, trestbps, chol, thalach, oldpeak]])

# apply 
input_data = imputer.transform(input_data)
input_data = scaler.transform(input_data)

# predict
if st.button("Dự đoán"):
    result = model.predict(input_data)[0]
    if result == 1:
        st.error("Bạn có nguy cơ mắc bệnh tim.")
    else:
        st.success("Bạn không có nguy cơ mắc bệnh tim.")