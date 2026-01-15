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
# KOLOM DATASET (HARUS SAMA DENGAN TRAINING)
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
    "Unemployment rate","Inflation rate","GDP"
]

# ==================================================
# TEMPLATE & DEMO DATA
# ==================================================
template_df = pd.DataFrame(columns=FEATURE_COLUMNS)

demo_df = pd.DataFrame(
    [[0]*len(FEATURE_COLUMNS)],
    columns=FEATURE_COLUMNS
)

# ==================================================
# TABS
# ==================================================
tab1, tab2 = st.tabs(["ℹ️ Informasi", "📊 Prediksi"])

# ==================================================
# TAB 1 — INFORMASI
# ==================================================
with tab1:
    st.title("🎓 Prediksi Dropout Mahasiswa")
    st.markdown("""
    Aplikasi ini memprediksi **Dropout / Graduate** mahasiswa
    menggunakan **model XGBoost** yang telah dilatih sebelumnya.

    **Fitur utama:**
    - Upload Excel
    - Validasi data otomatis
    - Prediksi batch
    - Grafik ringkasan
    - Download hasil
    """)

# ==================================================
# TAB 2 — PREDIKSI
# ==================================================
with tab2:
    st.title("📊 Prediksi Data Mahasiswa")

    st.markdown("""
    ### 📝 Petunjuk
    1. Download template Excel
    2. Isi data sesuai kolom
    3. Tidak boleh ada nilai kosong (NaN)
    4. Upload file lalu klik **Prediksi**
    """)

    colA, colB = st.columns(2)

    # ---------- TEMPLATE ----------
    with colA:
        buffer = BytesIO()
        template_df.to_excel(buffer, index=False)
        st.download_button(
            "⬇️ Download Template Excel",
            data=buffer.getvalue(),
            file_name="template_data_mahasiswa.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ---------- DEMO ----------
    with colB:
        demo_buffer = BytesIO()
        demo_df.to_excel(demo_buffer, index=False)
        st.download_button(
            "🧪 Download Demo Data",
            data=demo_buffer.getvalue(),
            file_name="demo_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in FEATURE_COLUMNS]

        if missing_cols or extra_cols:
            st.error("❌ Struktur kolom tidak sesuai")
            if missing_cols:
                st.warning(f"Kolom hilang: {missing_cols}")
            if extra_cols:
                st.warning(f"Kolom tidak dikenali: {extra_cols}")

        elif df.isnull().any().any():
            st.error("❌ Terdapat nilai kosong (NaN)")
            st.warning(df.columns[df.isnull().any()].tolist())

        else:
            st.success("✅ Data valid")

            if st.button("🔍 Jalankan Prediksi"):
                preds = model.predict(df)
                probs = model.predict_proba(df)

                df_result = df.copy()
                df_result["Prediction"] = preds
                df_result["Label"] = df_result["Prediction"].map(
                    {0: "Dropout", 1: "Graduate"}
                )
                df_result["Prob_Graduate"] = probs[:, 1]

                st.subheader("📄 Hasil Prediksi")
                st.dataframe(df_result)

                # ---------- GRAFIK ----------
                st.subheader("📊 Ringkasan Prediksi")
                summary = df_result["Label"].value_counts()

                fig, ax = plt.subplots()
                summary.plot(kind="bar", ax=ax)
                ax.set_ylabel("Jumlah Mahasiswa")
                st.pyplot(fig)

                # ---------- DOWNLOAD HASIL ----------
                out_buffer = BytesIO()
                df_result.to_excel(out_buffer, index=False)

                st.download_button(
                    "⬇️ Download Hasil (Excel)",
                    data=out_buffer.getvalue(),
                    file_name="hasil_prediksi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
