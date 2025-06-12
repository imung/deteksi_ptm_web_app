
# 🧠 Deteksi Dini Penyakit Tidak Menular (PTM) – Web App

Aplikasi web interaktif untuk mendeteksi risiko **Diabetes** dan **Hipertensi** berdasarkan input data pengguna seperti usia, tekanan darah, gula darah, indeks massa tubuh, dan riwayat keluarga.

Dibangun menggunakan:
- `Streamlit` (interface web)
- `scikit-learn` (model Random Forest)
- `joblib` (penyimpanan model dan encoder)

---

## 📦 Struktur Proyek

```
.
├── app.py                     # Aplikasi utama Streamlit
├── train_model.py            # Skrip pelatihan model
├── model_deteksi_ptm_rf.pkl  # Model ML terlatih
├── scaler.pkl                # Skaler fitur numerik
├── label_encoders.pkl        # Label encoder kolom kategorik
├── requirements.txt          # Daftar dependensi
```

---

## 🚀 Cara Menjalankan Web App

### 1. Clone / Unduh Proyek
```bash
git clone https://github.com/username/ptm-deteksi-app.git
cd ptm-deteksi-app
```

### 2. Install Dependency
Pastikan Python 3.7+ sudah terinstall. Lalu jalankan:

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

Akses di browser:
```
http://localhost:8501
```

---

## 🧪 Input yang Diminta Aplikasi

| Fitur                       | Keterangan                            |
|----------------------------|----------------------------------------|
| Usia                       | Umur individu (1–120)                  |
| Jenis Kelamin              | Laki-laki / Perempuan                  |
| Tekanan Darah Sistolik     | mmHg                                   |
| Tekanan Darah Diastolik    | mmHg                                   |
| Gula Darah Puasa           | mg/dL                                  |
| Indeks Massa Tubuh (IMT)   | dihitung manual                        |
| Merokok                    | Ya / Tidak                             |
| Aktivitas Fisik            | Rendah / Sedang / Tinggi               |
| Riwayat Keluarga Diabetes  | Ya / Tidak                             |
| Riwayat Keluarga Hipertensi| Ya / Tidak                             |

---

## 🎯 Output Prediksi

Model akan menampilkan:
- 🔵 **Normal**
- 🟡 **Diabetes**
- 🔴 **Hipertensi**

---

## ⚙️ Training Ulang Model (Opsional)

Jika ingin melatih ulang model:
```bash
python train_model.py
```

Akan menghasilkan:
- `model_deteksi_ptm_rf.pkl`
- `scaler.pkl`
- `label_encoders.pkl`

---

## 📃 Lisensi
MIT License © 2025
