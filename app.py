import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import matplotlib.pyplot as plt

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
# DEFINISI FITUR SESUAI TRAINING (36 FITUR)
# ==================================================
FEATURE_COLUMNS = [
    'Marital status',
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance\t',
    'Previous qualification',
    'Previous qualification (grade)',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Admission grade',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'Age at enrollment',
    'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

template_df = pd.DataFrame(columns=FEATURE_COLUMNS)

# ==================================================
# LAYOUT TAB
# ==================================================
tab1, tab2 = st.tabs(["🏠 Beranda", "📊 Prediksi"])

# ==================================================
# TAB 1 — BERANDA
# ==================================================
with tab1:
    st.title("🎓 Aplikasi Prediksi Dropout Mahasiswa")

    st.markdown("""
    Selamat datang di aplikasi **Prediksi Dropout Mahasiswa**.

    Aplikasi ini menggunakan algoritma **XGBoost** yang telah dilatih dengan data mahasiswa dari institusi pendidikan tinggi. Tujuan utama aplikasi ini adalah membantu:

    - 🏫 **Institusi pendidikan** dalam mendeteksi risiko mahasiswa dropout
    - 📊 **Analisis akademik** berdasarkan indikator historis
    - 🎯 **Pengambilan keputusan** untuk intervensi dini

    ---
    ## 🔍 Bagaimana aplikasi bekerja?

    1. **Input data mahasiswa** dalam format Excel sesuai template
    2. Sistem akan otomatis melakukan:
       - Validasi struktur kolom
       - Validasi nilai kosong
       - Konversi tipe data
    3. Model XGBoost akan menghitung prediksi:
       - **0 = Dropout**
       - **1 = Graduate**
    4. Hasil dapat dilihat langsung dan diunduh kembali dalam format Excel

    ---
    ## 📌 Keunggulan Aplikasi

    - Prediksi batch (banyak data sekaligus)
    - Template data otomatis
    - Validasi struktur & tipe data
    - Visualisasi hasil prediksi
    - Tampilan antarmuka interaktif

    Silakan lanjut ke tab **Prediksi** untuk mulai menggunakan aplikasi.
    """)

# ==================================================
# TAB 2 — PREDIKSI
# ==================================================
with tab2:

    st.title("📊 Prediksi Data Mahasiswa")

    st.markdown("""
    ### 📝 Petunjuk Penggunaan

    1. **Download template Excel** menggunakan tombol di bawah
    2. **Isi data mahasiswa** sesuai format dan urutan kolom
    3. Pastikan **tidak ada nilai kosong**
    4. Unggah file Excel ke aplikasi
    5. Klik tombol **Prediksi** untuk melihat hasilnya

    ---
    """)

    # Download template
    buf = BytesIO()
    template_df.to_excel(buf, index=False)

    st.download_button(
        "⬇️ Download Template Excel (36 kolom)",
        buf.getvalue(),
        "template_data_mahasiswa.xlsx"
    )

    st.divider()

    # Upload Excel
    uploaded = st.file_uploader(
        "📂 Upload file Excel",
        type=["xlsx", "xls"]
    )

    if uploaded:
        df = pd.read_excel(uploaded)
        st.write("Preview data:")
        st.dataframe(df.head())

        # VALIDASI KOLOM
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in FEATURE_COLUMNS]

        if missing:
            st.error(f"❌ Kolom hilang: {missing}")
            st.stop()

        if extra:
            st.warning(f"⚠️ Kolom tambahan diabaikan: {extra}")

        # URUTKAN SESUAI MODEL
        df = df[FEATURE_COLUMNS]

        # VALIDASI NILAI KOSONG
        if df.isnull().any().any():
            st.error("❌ Data mengandung nilai kosong. Mohon lengkapi sebelum upload.")
            st.stop()

        # KONVERSI NUMERIC
        try:
            df = df.apply(pd.to_numeric, errors="raise")
        except Exception as e:
            st.error(f"❌ Error konversi data ke numeric: {e}")
            st.stop()

        st.success("✅ Data valid dan siap diprediksi!")

        if st.button("🔍 Jalankan Prediksi"):
            X = df.values
            preds = model.predict(X)
            probs = model.predict_proba(X)

            df["Prediction"] = preds
            df["Label"] = df["Prediction"].map({0: "Dropout", 1: "Graduate"})
            df["Prob_Graduate"] = probs[:, 1]

            st.subheader("📄 Hasil Prediksi")
            st.dataframe(df)

            # GRAFIK
            fig, ax = plt.subplots()
            df["Label"].value_counts().plot(kind="bar", ax=ax)
            ax.set_ylabel("Jumlah Mahasiswa")
            st.pyplot(fig)

            # DOWNLOAD HASIL
            out = BytesIO()
            df.to_excel(out, index=False)

            st.download_button(
                "⬇️ Download Hasil Prediksi",
                out.getvalue(),
                "hasil_prediksi.xlsx"
            )
