import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
from datetime import datetime

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Dự đoán Bệnh Tim",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #c0392b 0%, #922b21 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(192,57,43,0.25);
}
.main-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
.main-header p  { margin: 0.5rem 0 0; opacity: 0.88; font-size: 1rem; }

.metric-card {
    background: white;
    border: 1px solid #f0f0f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #c0392b; }
.metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }

.info-box {
    background: #fef9f9;
    border-left: 4px solid #c0392b;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}

.result-positive {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: white;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(231,76,60,0.3);
}
.result-negative {
    background: linear-gradient(135deg, #27ae60, #1e8449);
    color: white;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 6px 24px rgba(39,174,96,0.3);
}
.result-positive h2, .result-negative h2 { margin: 0; font-size: 1.6rem; }
.result-positive p,  .result-negative p  { margin: 0.5rem 0 0; opacity: 0.9; }

.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
    border-bottom: 2px solid #e8e8e8;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

.input-note {
    font-size: 0.82rem;
    color: #888;
    font-style: italic;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #c0392b, #922b21);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Be Vietnam Pro', sans-serif;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    box-shadow: 0 4px 16px rgba(192,57,43,0.4);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOAD MODEL & DATA (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "heart_model.h5"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def load_data():
    data_path = "heart.csv"
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)

model_data = load_model()
df_raw = load_data()

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Tim Mạch AI")
    st.markdown("---")
    page = st.radio(
        "📌 Chọn trang",
        ["🏠 Trang chủ", "🔍 Kiểm tra sức khỏe"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; opacity:0.7; margin-top:1rem;'>
    ⚠️ <b>Lưu ý:</b> Đây là công cụ hỗ trợ tham khảo, không thay thế chẩn đoán y khoa.
    Hãy gặp bác sĩ để được tư vấn chính xác.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════
#  TRANG CHỦ
# ══════════════════════════════════════════
if page == "🏠 Trang chủ":

    st.markdown("""
    <div class="main-header">
        <h1>🫀 Phân tích Dữ liệu Bệnh Tim</h1>
        <p>UCI Heart Disease Dataset · 920 bệnh nhân · 14 đặc trưng y tế</p>
    </div>
    """, unsafe_allow_html=True)

    # ── GIỚI THIỆU DỮ LIỆU ──
    st.markdown("### 📊 Giới thiệu Bộ Dữ liệu")
    st.markdown("""
    <div class="info-box">
    Bộ dữ liệu <b>UCI Heart Disease</b> thu thập từ 4 cơ sở y tế (Cleveland, Hungary,
    Switzerland, VA Long Beach). Mỗi dòng tương ứng một bệnh nhân với 14 thông số
    lâm sàng và một nhãn chẩn đoán (<code>num</code>: 0 = không bệnh, 1–4 = mức độ bệnh tăng dần).
    </div>
    """, unsafe_allow_html=True)

    if df_raw is not None:
        df = df_raw.copy()
        df['fbs']  = df['fbs'].map({True: 1, False: 0, 1: 1, 0: 0})
        df['exang'] = df['exang'].map({True: 1, False: 0, 1: 1, 0: 0})
        df['has_disease'] = (df['num'] > 0).astype(int)

        # Tóm tắt số liệu
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{len(df)}</div>
                <div class="label">Tổng bệnh nhân</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <div class="value">14</div>
                <div class="label">Đặc trưng đầu vào</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            pct = df['has_disease'].mean()*100
            st.markdown(f"""<div class="metric-card">
                <div class="value">{pct:.0f}%</div>
                <div class="label">Tỉ lệ có bệnh</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            mean_age = df['age'].mean()
            st.markdown(f"""<div class="metric-card">
                <div class="value">{mean_age:.0f}</div>
                <div class="label">Tuổi trung bình</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BẢNG MÔ TẢ CỘT ──
        with st.expander("📋 Xem mô tả các cột dữ liệu"):
            cols_desc = pd.DataFrame({
                "Tên cột":    ["age","sex","cp","trestbps","chol","fbs","restecg","thalch","exang","oldpeak","slope","ca","thal","num"],
                "Mô tả":      ["Tuổi (năm)", "Giới tính (Male/Female)", "Loại đau ngực", "Huyết áp tâm thu (mmHg)",
                               "Cholesterol huyết thanh (mg/dL)", "Đường huyết đói > 120 mg/dL", "Kết quả điện tâm đồ",
                               "Nhịp tim tối đa (bpm)", "Đau ngực khi gắng sức", "Chênh lệch ST so với nghỉ",
                               "Độ dốc đoạn ST đỉnh", "Số mạch chính qua huỳnh quang", "Kết quả Thalassemia",
                               "Mức độ bệnh (0=không, 1-4=tăng dần)"],
                "Loại dữ liệu": ["Số nguyên","Chuỗi","Chuỗi","Số nguyên","Số nguyên","Bool","Chuỗi","Số nguyên","Bool","Số thực","Chuỗi","Số nguyên","Chuỗi","Số nguyên"]
            })
            st.dataframe(cols_desc, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BIỂU ĐỒ EDA ──
        st.markdown("### 📈 Phân tích Trực quan")
        plt.rcParams.update({'font.family': 'DejaVu Sans'})
        sns.set_style("whitegrid")

        # Row 1
        fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.5))
        fig1.patch.set_facecolor('#fafafa')

        # Chart 1: Phân phối nhãn
        ax = axes1[0]
        counts = df['num'].value_counts().sort_index()
        colors_bar = ['#27ae60','#f39c12','#e67e22','#e74c3c','#8e44ad']
        bars = ax.bar([f"Độ {i}" for i in counts.index], counts.values, color=colors_bar[:len(counts)], edgecolor='white', linewidth=1.5, width=0.6)
        ax.set_title('Phân phối Mức độ Bệnh Tim', fontweight='bold', fontsize=11)
        ax.set_ylabel('Số bệnh nhân')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4, str(val), ha='center', fontsize=9, fontweight='bold')
        ax.set_facecolor('#f8f9fa')

        # Chart 2: Tuổi vs bệnh
        ax = axes1[1]
        df_no  = df[df['has_disease'] == 0]['age']
        df_yes = df[df['has_disease'] == 1]['age']
        ax.hist(df_no,  bins=15, alpha=0.75, color='#27ae60', label='Không bệnh', edgecolor='white')
        ax.hist(df_yes, bins=15, alpha=0.75, color='#e74c3c', label='Có bệnh',    edgecolor='white')
        ax.set_title('Phân phối Tuổi theo Bệnh Tim', fontweight='bold', fontsize=11)
        ax.set_xlabel('Tuổi')
        ax.set_ylabel('Số bệnh nhân')
        ax.legend(fontsize=9)
        ax.axvline(df_no.mean(),  color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axvline(df_yes.mean(), color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_facecolor('#f8f9fa')

        # Chart 3: Giới tính
        ax = axes1[2]
        sex_tbl = pd.crosstab(df['sex'], df['has_disease'])
        sex_tbl.columns = ['Không bệnh', 'Có bệnh']
        sex_tbl.plot(kind='bar', ax=ax, color=['#27ae60','#e74c3c'],
                     edgecolor='white', rot=0, width=0.6)
        ax.set_title('Giới Tính và Bệnh Tim', fontweight='bold', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Số bệnh nhân')
        ax.legend(fontsize=9)
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout(pad=2)
        st.pyplot(fig1)
        plt.close()

        # Row 2
        fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.5))
        fig2.patch.set_facecolor('#fafafa')

        # Chart 4: Nhịp tim tối đa
        ax = axes2[0]
        ax.boxplot(
            [df[df['has_disease']==0]['thalch'].dropna(),
             df[df['has_disease']==1]['thalch'].dropna()],
            labels=['Không bệnh', 'Có bệnh'],
            patch_artist=True,
            boxprops=dict(facecolor='#aed6f1', alpha=0.8),
            medianprops=dict(color='#e74c3c', linewidth=2),
            flierprops=dict(marker='o', alpha=0.3)
        )
        ax.set_title('Nhịp Tim Tối Đa theo Bệnh Tim', fontweight='bold', fontsize=11)
        ax.set_ylabel('Nhịp tim tối đa (bpm)')
        ax.set_facecolor('#f8f9fa')

        # Chart 5: Loại đau ngực
        ax = axes2[1]
        cp_tbl = pd.crosstab(df['cp'], df['has_disease'])
        cp_tbl.columns = ['Không bệnh', 'Có bệnh']
        cp_pct = cp_tbl.div(cp_tbl.sum(axis=1), axis=0) * 100
        cp_pct.plot(kind='barh', ax=ax, color=['#27ae60','#e74c3c'],
                    edgecolor='white', width=0.6)
        ax.set_title('Loại Đau Ngực (% theo nhóm)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Tỉ lệ (%)')
        ax.legend(fontsize=9)
        ax.set_facecolor('#f8f9fa')

        # Chart 6: Oldpeak distribution
        ax = axes2[2]
        ax.hist(df[df['has_disease']==0]['oldpeak'].dropna(), bins=15,
                alpha=0.75, color='#27ae60', label='Không bệnh', edgecolor='white')
        ax.hist(df[df['has_disease']==1]['oldpeak'].dropna(), bins=15,
                alpha=0.75, color='#e74c3c', label='Có bệnh', edgecolor='white')
        ax.set_title('Chênh lệch ST (Oldpeak)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Giá trị ST Depression')
        ax.set_ylabel('Số bệnh nhân')
        ax.legend(fontsize=9)
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout(pad=2)
        st.pyplot(fig2)
        plt.close()

        # ── KẾT LUẬN ──
        st.markdown("### 💡 Kết luận từ Phân tích Dữ liệu")

        conclusions = [
            ("👴 Tuổi tác", "Bệnh nhân có bệnh tim thường lớn tuổi hơn (TB ~57 tuổi) so với nhóm khỏe mạnh (TB ~52 tuổi). Nguy cơ tăng đáng kể sau 50 tuổi."),
            ("👨 Giới tính", "Nam giới chiếm tỉ lệ mắc bệnh cao hơn nữ giới. Đây là xu hướng được ghi nhận trong nhiều nghiên cứu tim mạch."),
            ("💓 Nhịp tim tối đa", "Bệnh nhân có bệnh tim thường có nhịp tim tối đa thấp hơn – tim hoạt động kém hiệu quả hơn khi gắng sức."),
            ("⚡ Chênh lệch ST", "Giá trị oldpeak cao (chênh lệch ST lớn) liên quan chặt chẽ với nguy cơ mắc bệnh tim. Đây là một trong các đặc trưng quan trọng nhất."),
            ("😮 Loại đau ngực", "Nghịch lý: bệnh nhân 'không có triệu chứng điển hình' (asymptomatic) lại có tỉ lệ mắc bệnh cao nhất. Bệnh tim có thể tiến triển âm thầm."),
            ("📊 Dữ liệu không mất cân bằng nghiêm trọng", "Tỉ lệ có bệnh ~55%, không có bệnh ~45% – dữ liệu tương đối cân bằng, phù hợp để huấn luyện mô hình mà không cần kỹ thuật oversampling."),
        ]

        c1, c2 = st.columns(2)
        for i, (title, text) in enumerate(conclusions):
            col = c1 if i % 2 == 0 else c2
            with col:
                st.markdown(f"""
                <div style="background:white; border:1px solid #eee; border-radius:10px;
                            padding:1rem 1.25rem; margin-bottom:0.8rem;
                            box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <b style="color:#c0392b;">{title}</b>
                    <p style="margin:0.4rem 0 0; font-size:0.92rem; color:#444;">{text}</p>
                </div>
                """, unsafe_allow_html=True)

    # ── THÔNG SỐ MÔ HÌNH ──
    st.markdown("---")
    st.markdown("### 🤖 Kết quả Huấn luyện Mô hình")
    st.markdown("""
    <div class="info-box">
    Mô hình sử dụng: <b>Gradient Boosting Classifier</b> (200 cây quyết định, depth=4) 
    với tiền xử lý SimpleImputer (điền giá trị thiếu) và StandardScaler (chuẩn hóa).
    Bài toán được đưa về <b>phân loại nhị phân</b>: có bệnh tim hay không.
    </div>
    """, unsafe_allow_html=True)

    if model_data is not None:
        acc = model_data['accuracy']
        auc = model_data.get('auc', None)
        cr  = model_data['classification_report']
        fi  = model_data.get('feature_importance', None)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="value">{acc*100:.1f}%</div>
                <div class="label">Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            if auc:
                st.markdown(f"""<div class="metric-card">
                    <div class="value">{auc:.3f}</div>
                    <div class="label">ROC-AUC</div>
                </div>""", unsafe_allow_html=True)
        with m3:
            prec = cr.get('1', cr.get('weighted avg', {})).get('precision', 0)
            st.markdown(f"""<div class="metric-card">
                <div class="value">{prec:.3f}</div>
                <div class="label">Precision (Có bệnh)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            rec = cr.get('1', cr.get('weighted avg', {})).get('recall', 0)
            st.markdown(f"""<div class="metric-card">
                <div class="value">{rec:.3f}</div>
                <div class="label">Recall (Có bệnh)</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance chart
        if fi is not None:
            fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
            fi_top = fi.sort_values(ascending=True).tail(12)
            colors_fi = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(fi_top)))
            ax_fi.barh(range(len(fi_top)), fi_top.values, color=colors_fi, edgecolor='white')
            ax_fi.set_yticks(range(len(fi_top)))
            ax_fi.set_yticklabels(fi_top.index, fontsize=10)
            ax_fi.set_xlabel('Độ quan trọng (Feature Importance)')
            ax_fi.set_title('Top 12 Đặc trưng Quan trọng Nhất', fontweight='bold', fontsize=12)
            ax_fi.set_facecolor('#f8f9fa')
            fig_fi.patch.set_facecolor('#fafafa')
            plt.tight_layout()
            st.pyplot(fig_fi)
            plt.close()

        # Confusion matrix
        cm = model_data['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Không bệnh', 'Có bệnh'],
                    yticklabels=['Không bệnh', 'Có bệnh'],
                    linewidths=1, linecolor='white', ax=ax_cm,
                    annot_kws={'size': 14, 'weight': 'bold'})
        ax_cm.set_xlabel('Dự đoán')
        ax_cm.set_ylabel('Thực tế')
        ax_cm.set_title('Ma trận Nhầm lẫn (Confusion Matrix)', fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()

    else:
        st.warning("⚠️ Chưa tìm thấy file mô hình `heart_model.h5`. Vui lòng chạy notebook để huấn luyện mô hình trước.")


# ══════════════════════════════════════════
#  TRANG KIỂM TRA
# ══════════════════════════════════════════
elif page == "🔍 Kiểm tra sức khỏe":

    st.markdown("""
    <div class="main-header">
        <h1>🔍 Kiểm tra Nguy cơ Bệnh Tim</h1>
        <p>Nhập các thông số sức khỏe bên dưới để nhận kết quả dự đoán từ mô hình AI</p>
    </div>
    """, unsafe_allow_html=True)

    if model_data is None:
        st.error("❌ Chưa tìm thấy file mô hình. Vui lòng chạy notebook `classification.ipynb` trước để tạo `heart_model.h5`.")
        st.stop()

    model      = model_data['model']
    imputer    = model_data['imputer']
    scaler     = model_data['scaler']
    feat_cols  = model_data['feature_columns']

    # ── INPUT FORM ──
    st.markdown("### 📝 Thông tin cơ bản")
    st.markdown('<p class="input-note">Vui lòng điền đầy đủ các thông số bên dưới. Nếu không có số liệu, hãy để giá trị mặc định.</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">👤 Thông tin cá nhân</div>', unsafe_allow_html=True)

        st.markdown("**📅 Tuổi của bạn là bao nhiêu?**")
        age = st.number_input("Tuổi", min_value=20, max_value=100, value=55,
                               label_visibility="collapsed")

        st.markdown("**🧑 Giới tính**")
        sex_vi = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True,
                           label_visibility="collapsed")
        sex = "Male" if sex_vi == "Nam" else "Female"

        st.markdown("**💔 Bạn có cảm thấy đau ngực không?**")
        st.caption("Chọn mô tả gần đúng nhất với triệu chứng của bạn")
        cp_vi = st.selectbox("Đau ngực", [
            "Không đau ngực (không có triệu chứng)",
            "Đau nhói hoặc tức ngực thông thường (không do tim)",
            "Đau thắt ngực không điển hình (tức/nặng ngực thoáng qua)",
            "Đau thắt ngực điển hình (đau dữ dội, lan ra vai/tay)"
        ], label_visibility="collapsed")

        cp_map = {
            "Không đau ngực (không có triệu chứng)": "asymptomatic",
            "Đau nhói hoặc tức ngực thông thường (không do tim)": "non-anginal",
            "Đau thắt ngực không điển hình (tức/nặng ngực thoáng qua)": "atypical angina",
            "Đau thắt ngực điển hình (đau dữ dội, lan ra vai/tay)": "typical angina"
        }
        cp = cp_map[cp_vi]

    with col_b:
        st.markdown('<div class="section-header">🩺 Chỉ số lâm sàng</div>', unsafe_allow_html=True)

        st.markdown("**🩸 Huyết áp tâm thu (mmHg)**")
        st.caption("Đo khi nghỉ ngơi. Bình thường: 90–120 mmHg. Cao huyết áp: > 140 mmHg")
        trestbps = st.number_input("Huyết áp", min_value=80, max_value=220, value=130,
                                    label_visibility="collapsed")

        st.markdown("**🧪 Cholesterol huyết thanh (mg/dL)**")
        st.caption("Bình thường: < 200 mg/dL. Cao: > 240 mg/dL. Thường lấy từ xét nghiệm máu")
        chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=220,
                                 label_visibility="collapsed")

        st.markdown("**🍬 Đường huyết đói có cao không?**")
        st.caption("Đường huyết > 120 mg/dL sau khi nhịn ăn 8 tiếng (tiểu đường)")
        fbs_vi = st.radio("Đường huyết", ["Không (< 120 mg/dL)", "Có (≥ 120 mg/dL)"],
                           horizontal=True, label_visibility="collapsed")
        fbs = 1 if "Có" in fbs_vi else 0

        st.markdown("**💓 Nhịp tim tối đa đạt được (bpm)**")
        st.caption("Nhịp tim cao nhất đo được khi gắng sức (thường từ 100–180 bpm)")
        thalch = st.number_input("Nhịp tim tối đa", min_value=60, max_value=230, value=150,
                                   label_visibility="collapsed")

    # Row 2
    st.markdown("---")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">🏃 Kết quả Gắng sức</div>', unsafe_allow_html=True)

        st.markdown("**😣 Khi tập thể dục hoặc leo cầu thang, bạn có bị đau ngực không?**")
        exang_vi = st.radio("Đau ngực khi gắng sức", ["Không", "Có"], horizontal=True,
                             label_visibility="collapsed")
        exang = 1 if exang_vi == "Có" else 0

        st.markdown("**📉 Chênh lệch đoạn ST so với lúc nghỉ (oldpeak)**")
        st.caption("Chỉ số đo từ điện tâm đồ (ECG). Bình thường: 0–1. Bất thường: > 2")
        oldpeak = st.number_input("Chênh lệch ST", min_value=0.0, max_value=10.0,
                                   value=1.0, step=0.1, label_visibility="collapsed")

        st.markdown("**📈 Độ dốc đoạn ST khi gắng sức tối đa**")
        st.caption("Lấy từ kết quả điện tâm đồ (ECG) khi gắng sức")
        slope_vi = st.selectbox("Độ dốc ST", [
            "Đi lên (upsloping) – thường gặp ở người khỏe",
            "Bằng phẳng (flat) – đáng lo ngại",
            "Đi xuống (downsloping) – cần chú ý"
        ], label_visibility="collapsed")

        slope_map = {
            "Đi lên (upsloping) – thường gặp ở người khỏe": "upsloping",
            "Bằng phẳng (flat) – đáng lo ngại": "flat",
            "Đi xuống (downsloping) – cần chú ý": "downsloping"
        }
        slope = slope_map[slope_vi]

    with col_d:
        st.markdown('<div class="section-header">🔬 Xét nghiệm bổ sung</div>', unsafe_allow_html=True)

        st.markdown("**🫀 Kết quả điện tâm đồ (ECG) khi nghỉ**")
        st.caption("Nếu không có kết quả ECG, chọn 'Bình thường'")
        restecg_vi = st.selectbox("Kết quả ECG", [
            "Bình thường",
            "Có bất thường sóng ST-T (đảo ngược hoặc chênh ST)",
            "Phì đại thất trái (LV hypertrophy)"
        ], label_visibility="collapsed")

        restecg_map = {
            "Bình thường": "normal",
            "Có bất thường sóng ST-T (đảo ngược hoặc chênh ST)": "st-t abnormality",
            "Phì đại thất trái (LV hypertrophy)": "lv hypertrophy"
        }
        restecg = restecg_map[restecg_vi]

        st.markdown("**🔵 Số mạch máu chính nhìn thấy qua chụp X-quang**")
        st.caption("Từ 0 đến 3. Nếu không có kết quả, giữ là 0")
        ca = st.selectbox("Số mạch máu", [0, 1, 2, 3], label_visibility="collapsed")

        st.markdown("**🧬 Kết quả xét nghiệm Thalassemia**")
        st.caption("Xét nghiệm tưới máu cơ tim. Nếu không có kết quả, chọn 'Bình thường'")
        thal_vi = st.selectbox("Thalassemia", [
            "Bình thường",
            "Khiếm khuyết cố định (fixed defect) – vùng tim không nhận máu)",
            "Khiếm khuyết đảo ngược (reversable defect) – giảm máu khi gắng sức"
        ], label_visibility="collapsed")

        thal_map = {
            "Bình thường": "normal",
            "Khiếm khuyết cố định (fixed defect) – vùng tim không nhận máu)": "fixed defect",
            "Khiếm khuyết đảo ngược (reversable defect) – giảm máu khi gắng sức": "reversable defect"
        }
        thal = thal_map[thal_vi]

    # ── NÚT DỰ ĐOÁN ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("🔍 Kiểm tra nguy cơ bệnh tim"):

        # --- Encode input ---
        input_dict = {
            'age': age, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'thalch': thalch, 'exang': exang,
            'oldpeak': oldpeak, 'ca': ca,
            'sex': sex, 'cp': cp, 'restecg': restecg,
            'slope': slope, 'thal': thal
        }

        df_in = pd.DataFrame([input_dict])
        cat_c = ['sex', 'cp', 'restecg', 'slope', 'thal']
        df_enc = pd.get_dummies(df_in, columns=cat_c)
        df_enc = df_enc.reindex(columns=feat_cols, fill_value=0)

        X_imp = imputer.transform(df_enc)
        X_sc  = scaler.transform(X_imp)

        pred  = model.predict(X_sc)[0]
        prob  = model.predict_proba(X_sc)[0][1]

        # Estimate severity level from probability
        if pred == 0:
            level = 0
        elif prob < 0.60:
            level = 1
        elif prob < 0.75:
            level = 2
        elif prob < 0.88:
            level = 3
        else:
            level = 4

        level_desc = {
            0: "Không có bệnh tim",
            1: "Bệnh nhẹ – Cần theo dõi định kỳ",
            2: "Bệnh vừa – Nên khám chuyên khoa tim mạch",
            3: "Bệnh nặng – Cần can thiệp y tế sớm",
            4: "Bệnh rất nặng – Cần điều trị khẩn cấp"
        }

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Display result ---
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            if pred == 0:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ Không phát hiện nguy cơ</h2>
                    <p style="font-size:1.1rem;">Xác suất mắc bệnh tim: <b>{prob*100:.1f}%</b></p>
                    <p>Mức độ ước tính: <b>Cấp độ {level} / 4</b> — {level_desc[level]}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ Phát hiện nguy cơ bệnh tim</h2>
                    <p style="font-size:1.1rem;">Xác suất mắc bệnh tim: <b>{prob*100:.1f}%</b></p>
                    <p>Mức độ ước tính: <b>Cấp độ {level} / 4</b> — {level_desc[level]}</p>
                </div>
                """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        fig_prob, ax_prob = plt.subplots(figsize=(8, 1.2))
        bar_color = '#e74c3c' if pred == 1 else '#27ae60'
        ax_prob.barh(['Xác suất'], [prob], color=bar_color, alpha=0.85, height=0.5)
        ax_prob.barh(['Xác suất'], [1 - prob], left=[prob], color='#ecf0f1', alpha=0.5, height=0.5)
        ax_prob.axvline(0.5, color='#7f8c8d', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_prob.set_xticklabels(['0%', '25%', '50%\n(ngưỡng)', '75%', '100%'])
        ax_prob.text(prob, 0, f'{prob*100:.1f}%', ha='center', va='bottom',
                      fontweight='bold', fontsize=11)
        ax_prob.set_facecolor('#fafafa')
        fig_prob.patch.set_facecolor('#fafafa')
        ax_prob.tick_params(left=False, labelleft=False)
        plt.tight_layout()

        _, pc, _ = st.columns([1, 2, 1])
        with pc:
            st.pyplot(fig_prob)
        plt.close()

        # Severity scale
        st.markdown("<br>", unsafe_allow_html=True)
        level_colors = ['#27ae60','#f1c40f','#e67e22','#e74c3c','#8e44ad']
        level_labels = ['Độ 0\nKhông bệnh','Độ 1\nNhẹ','Độ 2\nVừa','Độ 3\nNặng','Độ 4\nRất nặng']

        fig_lv, ax_lv = plt.subplots(figsize=(8, 1.8))
        for i, (c, lbl) in enumerate(zip(level_colors, level_labels)):
            alpha = 1.0 if i == level else 0.25
            rect = plt.Rectangle((i, 0), 0.92, 1, color=c, alpha=alpha)
            ax_lv.add_patch(rect)
            ax_lv.text(i + 0.46, 0.5, lbl, ha='center', va='center',
                       fontsize=8.5, fontweight='bold' if i == level else 'normal',
                       color='white' if i == level else '#777')
            if i == level:
                ax_lv.text(i + 0.46, 1.1, '▼', ha='center', va='bottom',
                           fontsize=10, color=c)
        ax_lv.set_xlim(0, 5)
        ax_lv.set_ylim(-0.1, 1.5)
        ax_lv.axis('off')
        ax_lv.set_title(f'Mức độ bệnh ước tính: Cấp độ {level}', fontsize=11, fontweight='bold')
        fig_lv.patch.set_facecolor('#fafafa')
        plt.tight_layout()

        _, lc, _ = st.columns([1, 2, 1])
        with lc:
            st.pyplot(fig_lv)
        plt.close()

        # ── LỜI KHUYÊN ──
        st.markdown("<br>", unsafe_allow_html=True)
        advice_map = {
            0: "✅ Kết quả cho thấy không có dấu hiệu đáng lo ngại. Hãy duy trì lối sống lành mạnh: ăn uống cân bằng, tập thể dục đều đặn, kiểm tra sức khỏe định kỳ mỗi năm.",
            1: "⚡ Có một số yếu tố nguy cơ nhẹ. Nên theo dõi huyết áp và cholesterol thường xuyên, giảm muối, tăng cường vận động, và kiểm tra tim mạch trong vòng 6 tháng tới.",
            2: "🔶 Nguy cơ ở mức vừa. Hãy đặt lịch khám chuyên khoa tim mạch sớm. Điều chỉnh chế độ ăn, hạn chế mỡ bão hòa và muối, không hút thuốc, kiểm soát căng thẳng.",
            3: "🔴 Nguy cơ cao. Cần gặp bác sĩ chuyên khoa tim mạch trong thời gian sớm nhất để được thăm khám và chỉ định xét nghiệm chuyên sâu (siêu âm tim, điện tâm đồ gắng sức).",
            4: "🆘 Nguy cơ rất cao. Vui lòng đến cơ sở y tế ngay để được chẩn đoán và điều trị kịp thời. Đây là ưu tiên y tế khẩn cấp."
        }

        st.info(f"**💬 Lời khuyên:** {advice_map[level]}")
        st.warning("⚠️ **Lưu ý quan trọng:** Kết quả này chỉ mang tính tham khảo từ mô hình AI, không thay thế chẩn đoán của bác sĩ. Hãy luôn tham khảo ý kiến chuyên gia y tế.")

        # ── LƯU DỮ LIỆU ──
        user_record = {
            'timestamp':   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age':         age,
            'sex':         sex_vi,
            'cp':          cp_vi,
            'trestbps':    trestbps,
            'chol':        chol,
            'fbs':         fbs_vi,
            'restecg':     restecg_vi,
            'thalch':      thalch,
            'exang':       exang_vi,
            'oldpeak':     oldpeak,
            'slope':       slope_vi,
            'ca':          ca,
            'thal':        thal_vi,
            'prediction':  int(pred),
            'probability': round(float(prob), 4),
            'level':       level
        }

        csv_path = "../data/userinput.csv"
        df_record = pd.DataFrame([user_record])
        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path)
            df_out = pd.concat([df_existing, df_record], ignore_index=True)
        else:
            df_out = df_record
        df_out.to_csv(csv_path, index=False)

        st.success(f"✅ Đã lưu kết quả vào `userinput.csv` (tổng {len(df_out)} lượt kiểm tra)")
