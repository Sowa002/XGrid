import streamlit as st
import pandas as pd
import pickle

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    layout="wide"
)

# ==================================================
# LOAD MODEL (TANPA TRAINING)
# ==================================================
@st.cache_resource
def load_model():
    with open("XGBoost_dropout_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# ==================================================
# DAFTAR KOLOM DATASET (WAJIB SAMA DENGAN TRAINING)
# ==================================================
FEATURE_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

# ==================================================
# TEMPLATE EXCEL
# ==================================================
template_df = pd.DataFrame(columns=FEATURE_COLUMNS)

# ==================================================
# TABS
# ==================================================
tab1, tab2 = st.tabs([
    "ℹ️ Informasi Aplikasi",
    "📊 Prediksi Data Mahasiswa"
])

# ==================================================
# TAB 1 — INFORMASI
# ==================================================
with tab1:
    st.title("🎓 Prediksi Dropout Mahasiswa")

    st.markdown("""
    ### 📌 Deskripsi
    Aplikasi ini digunakan untuk **memprediksi kemungkinan mahasiswa mengalami
    dropout atau graduate** menggunakan **model Machine Learning XGBoost**.

    Model telah **dilatih sebelumnya**, dan aplikasi ini **hanya melakukan prediksi**
    berdasarkan data yang diunggah pengguna.

    ### 🧠 Model
    - Algoritma: **XGBoost Classifier**
    - Output:
        - `0` → **Dropout**
        - `1` → **Graduate**

    ### 🎯 Tujuan
    - Deteksi dini risiko dropout mahasiswa
    - Mendukung pengambilan keputusan berbasis data

    ---
    📘 *Aplikasi ini dibuat untuk keperluan penelitian / tugas akhir.*
    """)

# ==================================================
# TAB 2 — PREDIKSI
# ==================================================
with tab2:
    st.title("📊 Prediksi Data Mahasiswa")

    # --------------------------------------------------
    # PETUNJUK
    # --------------------------------------------------
    st.markdown("""
    ### 📝 Petunjuk Penggunaan
    1. Unduh **template Excel** yang disediakan
    2. Isi data mahasiswa sesuai kolom yang tersedia
    3. Pastikan **tidak ada nilai kosong (NaN)**
    4. Upload file Excel (.xlsx / .xls)
    5. Klik **Jalankan Prediksi**

    ⚠️ **Nama dan jumlah kolom HARUS sama persis dengan template**
    """)

    # --------------------------------------------------
    # DOWNLOAD TEMPLATE
    # --------------------------------------------------
    template_bytes = template_df.to_excel(
        index=False,
        engine="openpyxl"
    )

    st.download_button(
        label="⬇️ Download Template Excel",
        data=template_bytes,
        file_name="template_data_mahasiswa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    # --------------------------------------------------
    # UPLOAD FILE
    # --------------------------------------------------
    uploaded_file = st.file_uploader(
        "📂 Upload file Excel",
        type=["xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            st.subheader("📄 Preview Data")
            st.dataframe(df.head())

            # --------------------------------------------------
            # VALIDASI NAMA KOLOM
            # --------------------------------------------------
            uploaded_cols = list(df.columns)
            expected_cols = FEATURE_COLUMNS

            missing_cols = [c for c in expected_cols if c not in uploaded_cols]
            extra_cols = [c for c in uploaded_cols if c not in expected_cols]

            if missing_cols or extra_cols:
                st.error("❌ Struktur kolom tidak sesuai dengan template")

                if missing_cols:
                    st.warning(f"Kolom yang HILANG: {missing_cols}")

                if extra_cols:
                    st.warning(f"Kolom yang TIDAK DIKENAL: {extra_cols}")

            # --------------------------------------------------
            # VALIDASI MISSING VALUE
            # --------------------------------------------------
            elif df.isnull().any().any():
                na_cols = df.columns[df.isnull().any()].tolist()
                st.error("❌ Terdapat nilai kosong (NaN) pada data")
                st.warning(f"Kolom bermasalah: {na_cols}")

            else:
                st.success("✅ Data valid dan siap diprediksi")

                if st.button("🔍 Jalankan Prediksi"):
                    preds = model.predict(df)
                    probs = model.predict_proba(df)

                    df_result = df.copy()
                    df_result["Prediction"] = preds
                    df_result["Prediction_Label"] = df_result["Prediction"].map(
                        {0: "Dropout", 1: "Graduate"}
                    )
                    df_result["Probability_Dropout"] = probs[:, 0]
                    df_result["Probability_Graduate"] = probs[:, 1]

                    st.subheader("✅ Hasil Prediksi")
                    st.dataframe(df_result)

                    # --------------------------------------------------
                    # DOWNLOAD HASIL
                    # --------------------------------------------------
                    output_csv = df_result.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Hasil Prediksi",
                        data=output_csv,
                        file_name="hasil_prediksi_dropout.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file:\n\n{e}")
