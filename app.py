"""
app.py
Web App Streamlit - Deteksi Dini Diabetes
UAS Kecerdasan Buatan

Cara jalankan lokal:
    streamlit run app.py
"""
import streamlit as st
import numpy as np
import joblib

# ── Konfigurasi halaman ──────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Dini Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ── Load model & scaler (cached supaya tidak reload tiap interaksi) ─
@st.cache_resource
def load_model():
    model = pickle.load(open("best_model.pkl", "rb"))
    data = pickle.load(open("preprocessed_data.pkl", "rb"))
    scaler = data['scaler']
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Header ────────────────────────────────────────────────────────
st.title("🩺 Deteksi Dini Diabetes")
st.markdown("""
Aplikasi ini menggunakan model **Ensemble (XGBoost + Random Forest)** yang dioptimasi
dengan **Bayesian Optimization** dan **SMOTE** untuk memprediksi risiko diabetes
berdasarkan data klinis pasien.

*UAS Kecerdasan Buatan — bukan alat diagnosis medis resmi, hanya untuk keperluan akademik.*
""")

if not model_loaded:
    st.error(
        "⚠️ Model belum ditemukan. Pastikan file `outputs/best_model.pkl` dan "
        "`outputs/preprocessed_data.pkl` sudah ada di repository (hasil training dari "
        "notebook Colab)."
    )
    st.stop()

st.divider()

# ── Form input ────────────────────────────────────────────────────
st.subheader("📋 Masukkan Data Klinis Pasien")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=2, step=1)
    glucose = st.number_input("Kadar Glukosa (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Ketebalan Lipatan Kulit (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Kadar Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=28.5, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=35, step=1)

st.divider()

# ── Prediksi ──────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    X_new = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                        insulin, bmi, diabetes_pedigree, age]])
    X_new_scaled = scaler.transform(X_new)

    prediction = model.predict(X_new_scaled)[0]
    probability = model.predict_proba(X_new_scaled)[0]

    label = "Diabetik" if prediction == 1 else "Non-Diabetik"
    confidence = probability[prediction] * 100

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(f"**Hasil: {label}** (Tingkat keyakinan model: {confidence:.2f}%)")
        st.warning(
            "Model mendeteksi indikasi risiko diabetes. Ini **bukan diagnosis medis** — "
            "silakan konsultasikan ke dokter atau tenaga kesehatan untuk pemeriksaan lebih lanjut."
        )
    else:
        st.success(f"**Hasil: {label}** (Tingkat keyakinan model: {confidence:.2f}%)")
        st.info(
            "Model tidak mendeteksi indikasi risiko diabetes berdasarkan data yang dimasukkan. "
            "Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Probabilitas Non-Diabetik", f"{probability[0]*100:.2f}%")
    with col_b:
        st.metric("Probabilitas Diabetik", f"{probability[1]*100:.2f}%")

    st.progress(float(probability[1]))

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Voting Ensemble (XGBoost + Random Forest) · SMOTE untuk balancing kelas · "
    "Hyperparameter tuning dengan Optuna (Bayesian Optimization) · "
    "Dataset: Pima Indians Diabetes Database"
)
