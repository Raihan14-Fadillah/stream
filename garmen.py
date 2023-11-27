import streamlit as st
import pickle
import pandas as pd

# Memuat model scaler
with open("scaler_model.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Memuat model Random Forest dengan akurasi terbaik
with open("best_rf_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

st.title("Aplikasi Prediksi Produktivitas Karyawan")

# Input dari pengguna
st.header("Masukkan Nilai Fitur")

# Definisikan input untuk masing-masing fitur
smv = st.number_input("SMV (Nilai Menit Standar):", value=0.0, step=0.1, format="%.1f", help="Nilai menit standar yang dialokasikan untuk tugas.")
wip = st.number_input("Work in Progress (WIP):", value=0, step=1, help="Jumlah barang setengah jadi atau belum selesai dalam produksi.")
over_time = st.number_input("Over Time (Lembur) (dalam menit):", value=0, step=30, help="Jumlah waktu tambahan yang dihabiskan dalam produksi di atas waktu kerja reguler dalam menit.")
incentive = st.number_input("Insentif (dalam BDT):", value=0, step=100, help="Jumlah insentif finansial yang diberikan sebagai motivasi untuk meningkatkan produktivitas.")
idle_time = st.number_input("Waktu Tidak Efisien (dalam menit):", value=0, step=30, help="Jumlah waktu ketika produksi terhenti atau tidak efisien karena berbagai alasan.")
idle_men = st.number_input("Pekerja Tidak Efisien (Jumlah Pekerja):", value=0, step=1, help="Jumlah pekerja yang menjadi tidak efisien atau tidak dapat bekerja selama produksi terhenti atau tidak efisien.")
no_of_workers = st.number_input("Jumlah Pekerja:", value=0, step=1, help="Jumlah total pekerja yang bekerja selama periode produksi.")

# Tombol prediksi
if st.button("Prediksi Produktivitas"):
    # Menormalisasi input pengguna
    input_data = {
        'smv': smv,
        'wip': wip,
        'over_time': over_time,
        'incentive': incentive,
        'idle_time': idle_time,
        'idle_men': idle_men,
        'no_of_workers': no_of_workers
    }
    input_df = pd.DataFrame([input_data])
    X_normalized = scaler.transform(input_df)

    # Melakukan prediksi dengan model Random Forest
    prediction = rf_model.predict(X_normalized)

    # Konversi label produktivitas ke Bahasa Indonesia
    label_produk = ''
    if prediction[0] == 'High':
        label_produk = 'Tinggi'
    elif prediction[0] == 'Medium':
        label_produk = 'Sedang'
    elif prediction[0] == 'Low':
        label_produk = 'Rendah'

    st.write(f"Hasil Prediksi Produktivitas: {label_produk}")

# Menampilkan grafik dari semua inputan
st.subheader("Grafik Input Pengguna")
data = pd.DataFrame({'Fitur': ['SMV', 'WIP', 'Over Time', 'Insentif', 'Waktu Tidak Efisien', 'Pekerja Tidak Efisien', 'Jumlah Pekerja'],
                     'Nilai': [smv, wip, over_time, incentive, idle_time, idle_men, no_of_workers]})
st.bar_chart(data.set_index('Fitur'))