import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("XGBoost_dropout_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["ℹ️ Informasi Aplikasi", "📊 Prediksi Data"])

# =========================
# TAB 1 — INFORMASI
# =========================
with tab1:
    st.title("🎓 Prediksi Dropout Mahasiswa")

    st.markdown("""
    ### 📌 Tentang Aplikasi
    Aplikasi ini digunakan untuk **memprediksi kemungkinan mahasiswa
    mengalami dropout atau graduate** menggunakan model **XGBoost**.

    ### 🧠 Model
    - Algoritma: **XGBoost Classifier**
    - Output:
        - `0` → Dropout  
        - `1` → Graduate

    ### 📂 Cara Menggunakan
    1. Masuk ke tab **Prediksi Data**
    2. Upload file **Excel (.xlsx) atau CSV (.csv)**
    3. Pastikan kolom sesuai dengan dataset training
    4. Klik **Prediksi**
    5. Download hasil prediksi

    ---
    ✨ Aplikasi ini dibuat sebagai bagian dari penelitian / skripsi.
    """)

# =========================
# TAB 2 — PREDIKSI
# =========================
with tab2:
    st.title("📊 Prediksi Data Mahasiswa")

    uploaded_file = st.file_uploader(
        "Upload file Excel atau CSV",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        # =========================
        # BACA FILE
        # =========================
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("📄 Preview Data")
        st.dataframe(df.head())

        # =========================
        # CEK KOLOM
        # =========================
        required_features = model.n_features_in_

        st.info(f"Model membutuhkan {required_features} fitur")

        if df.shape[1] != required_features:
            st.error(
                f"Jumlah kolom tidak sesuai! "
                f"(Dataset: {df.shape[1]} kolom)"
            )
        else:
            if st.button("🔍 Jalankan Prediksi"):
                preds = model.predict(df)
                probs = model.predict_proba(df)

                df["Prediction"] = preds
                df["Probability_Graduate"] = probs[:, 1]
                df["Probability_Dropout"] = probs[:, 0]

                st.subheader("✅ Hasil Prediksi")
                st.dataframe(df)

                # =========================
                # DOWNLOAD
                # =========================
                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    csv,
                    "hasil_prediksi_dropout.csv",
                    "text/csv"
                )
