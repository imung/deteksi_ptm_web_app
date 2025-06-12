import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("C:/ptm/model_deteksi_ptm_rf.pkl")

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Sistem Deteksi Dini Penyakit Tidak Menular (PTM)")

st.write("Masukkan data individu untuk memprediksi risiko Diabetes atau Hipertensi")

# User inputs
usia = st.number_input("Usia", min_value=1, max_value=120, value=35)
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
tekanan_sistolik = st.number_input("Tekanan Darah Sistolik", value=120)
tekanan_diastolik = st.number_input("Tekanan Darah Diastolik", value=80)
gula_darah = st.number_input("Gula Darah Puasa", value=90)
imt = st.number_input("Indeks Massa Tubuh (IMT)", value=22.0)
merokok = st.selectbox("Apakah Merokok?", ["Ya", "Tidak"])
aktivitas = st.selectbox("Tingkat Aktivitas Fisik", ["Rendah", "Sedang", "Tinggi"])
riwayat_dm = st.selectbox("Riwayat Keluarga Diabetes", ["Ya", "Tidak"])
riwayat_ht = st.selectbox("Riwayat Keluarga Hipertensi", ["Ya", "Tidak"])

# Preprocessing input
def preprocess_input():
    input_data = pd.DataFrame({
        "Usia": [usia],
        "Jenis_Kelamin": [jenis_kelamin],
        "Tekanan_Darah_Sistolik": [tekanan_sistolik],
        "Tekanan_Darah_Diastolik": [tekanan_diastolik],
        "Gula_Darah_Puasa": [gula_darah],
        "IMT": [imt],
        "Merokok": [merokok],
        "Aktivitas_Fisik": [aktivitas],
        "Riwayat_Keluarga_Diabetes": [riwayat_dm],
        "Riwayat_Keluarga_Hipertensi": [riwayat_ht]
    })
    for col in label_encoders:
        if col in input_data.columns:
            input_data[col] = label_encoders[col].transform(input_data[col])
    input_scaled = scaler.transform(input_data)
    return input_scaled

# Predict button
if st.button("Prediksi Risiko"):
    X_input = preprocess_input()
    pred = model.predict(X_input)[0]
    diagnosis = label_encoders["Diagnosis"].inverse_transform([pred])[0]
    st.success(f"Hasil Prediksi: {diagnosis}")
