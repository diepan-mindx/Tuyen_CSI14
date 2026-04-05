# import
import streamlit as st
import numpy as np
import joblib
from savedata import save_data

# load model
data = joblib.load("heart_model.h5")
model = data["model"]
imputer = data["imputer"]
scaler = data["scaler"]

st.title("Dự đoán bệnh tim")
st.write("Nhập thông tin của bạn để dự đoán nguy cơ mắc bệnh tim.")

# ===== INPUT =====
age = st.number_input("Tuổi", 20, 100, 50)
sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
dataset = 1

cp = st.selectbox("Loại đau ngực", [
    "Không đau ngực",
    "Đau ngực không điển hình",
    "Đau ngực điển hình",
    "Đau ngực khác"
])

trestbps = st.number_input("Huyết áp", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)

fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl", ["Có", "Không"])

restecg = st.selectbox("Kết quả điện tâm đồ", [
    "Bình thường",
    "Có bất thường",
    "Có sóng Q"
])

thalach = st.number_input("Nhịp tim tối đa", 60, 220, 150)

exang = st.selectbox("Có đau ngực khi tập thể dục không?", ["Có", "Không"])

oldpeak = st.number_input("Giảm ST", 0.0, 10.0, 1.0)

slope = st.selectbox("Độ dốc của đoạn ST", [
    "Lên",
    "Bằng phẳng",
    "Xuống"
])

ca = st.number_input("Số lượng mạch chính", 0, 3, 0)

thal = st.selectbox("Thalassemia", [
    "Bình thường",
    "Có bất thường cố định",
    "Có bất thường đảo ngược"
])

# ===== ENCODE =====
def encode_inputs():
    sex_val = 1 if sex == "Nam" else 0

    cp_mapping = {
        "Không đau ngực": 0,
        "Đau ngực không điển hình": 1,
        "Đau ngực điển hình": 2,
        "Đau ngực khác": 3
    }

    fbs_val = 1 if fbs == "Có" else 0

    restecg_mapping = {
        "Bình thường": 0,
        "Có bất thường": 1,
        "Có sóng Q": 2
    }

    exang_val = 1 if exang == "Có" else 0

    slope_mapping = {
        "Lên": 0,
        "Bằng phẳng": 1,
        "Xuống": 2
    }

    thal_mapping = {
        "Bình thường": 1,
        "Có bất thường cố định": 2,
        "Có bất thường đảo ngược": 3
    }

    return np.array([[
        age,
        sex_val,
        dataset,
        cp_mapping[cp],
        trestbps,
        chol,
        fbs_val,
        restecg_mapping[restecg],
        thalach,
        exang_val,
        oldpeak,
        slope_mapping[slope],
        ca,
        thal_mapping[thal]
    ]])

# ===== SAVE DATA =====
if st.button("Lưu dữ liệu"):
    encoded = encode_inputs()[0]
    save_data(*encoded, 0)
    st.success("Đã lưu dữ liệu!")

# ===== PREDICT =====
if st.button("Dự đoán"):
    input_data = encode_inputs()

    # xử lý dữ liệu
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    # dự đoán
    result = model.predict(input_data)[0]

    if result == 1:
        st.error("Bạn có nguy cơ mắc bệnh tim.")
    else:
        st.success("Bạn không có nguy cơ mắc bệnh tim.")