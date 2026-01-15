import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import matplotlib.pyplot as plt

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    with open("XGBoost_dropout_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# ==================================================
# KOLOM FITUR (URUTAN WAJIB BENAR)
# ==================================================
FEATURE_COLUMNS = [
    "Marital status","Application mode","Application order","Course",
    "Daytime/evening attendance","Previous qualification",
    "Previous qualification (grade)","Admission grade","Displaced",
    "Educational special needs","Debtor","Tuition fees up to date","Gender",
    "Scholarship holder","Age at enrollment","International",
    "Curricular units 1st sem (credited)","Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)","Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)","Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)","Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate","Inflationation rate","GDP"
]

# ==================================================
# TEMPLATE & DEMO
# ==================================================
template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
demo_df = pd.DataFrame([[0]*len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)

# ==================================================
# TABS
# ==================================================
tab1, tab2 = st.tabs(["ℹ️ Informasi", "📊 Prediksi"])

with tab1:
    st.title("🎓 Prediksi Dropout Mahasiswa")
    st.markdown("""
    Aplikasi ini memprediksi status mahasiswa
    menggunakan **model XGBoost** yang telah dilatih sebelumnya.
    """)

with tab2:
    st.title("📊 Prediksi Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        buf = BytesIO()
        template_df.to_excel(buf, index=False)
        st.download_button(
            "⬇️ Download Template",
            data=buf.getvalue(),
            file_name="template_data_mahasiswa.xlsx"
        )

    with col2:
        demo_buf = BytesIO()
        demo_df.to_excel(demo_buf, index=False)
        st.download_button(
            "🧪 Download Demo Data",
            data=demo_buf.getvalue(),
            file_name="demo_data.xlsx"
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload file Excel",
        type=["xlsx", "xls"]
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.dataframe(df.head())

        # ---------- VALIDASI ----------
        if set(df.columns) != set(FEATURE_COLUMNS):
            st.error("❌ Nama kolom tidak sesuai template")
            st.stop()

        if df.isnull().any().any():
            st.error("❌ Data mengandung nilai kosong (NaN)")
            st.stop()

        st.success("✅ Data valid")

        if st.button("🔍 Jalankan Prediksi"):
            # 🔐 PAKSA URUTAN
            df = df[FEATURE_COLUMNS]

            # 🔥 FIX UTAMA ADA DI SINI
            preds = model.predict(df, validate_features=False)
            probs = model.predict_proba(df, validate_features=False)

            df_result = df.copy()
            df_result["Prediction"] = preds
            df_result["Label"] = df_result["Prediction"].map(
                {0: "Dropout", 1: "Graduate"}
            )
            df_result["Prob_Graduate"] = probs[:, 1]

            st.dataframe(df_result)

            # ---------- GRAFIK ----------
            fig, ax = plt.subplots()
            df_result["Label"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

            # ---------- DOWNLOAD ----------
            out = BytesIO()
            df_result.to_excel(out, index=False)
            st.download_button(
                "⬇️ Download Hasil",
                data=out.getvalue(),
                file_name="hasil_prediksi.xlsx"
            )
