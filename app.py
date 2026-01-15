import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL SAJA
# =========================
@st.cache_resource
def load_model():
    with open("XGBoost_dropout_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🎓 Prediksi Dropout Mahasiswa")

# =========================
# AMBIL NAMA FITUR DARI MODEL
# =========================
feature_names = model.feature_names_in_

st.subheader("📋 Input Data Mahasiswa")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

st.dataframe(input_df)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success(f"🎓 Graduate (Probabilitas: {probability[0][1]:.2f})")
    else:
        st.error(f"⚠️ Dropout (Probabilitas: {probability[0][0]:.2f})")
