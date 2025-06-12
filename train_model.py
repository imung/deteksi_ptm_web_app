import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Buat dataset dummy (contoh 100 data)
np.random.seed(42)
df = pd.DataFrame({
    'Usia': np.random.randint(20, 70, 100),
    'Jenis_Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], 100),
    'Tekanan_Darah_Sistolik': np.random.randint(100, 180, 100),
    'Tekanan_Darah_Diastolik': np.random.randint(60, 120, 100),
    'Gula_Darah_Puasa': np.random.randint(70, 250, 100),
    'IMT': np.round(np.random.uniform(18.5, 35, 100), 1),
    'Merokok': np.random.choice(['Ya', 'Tidak'], 100),
    'Aktivitas_Fisik': np.random.choice(['Rendah', 'Sedang', 'Tinggi'], 100),
    'Riwayat_Keluarga_Diabetes': np.random.choice(['Ya', 'Tidak'], 100),
    'Riwayat_Keluarga_Hipertensi': np.random.choice(['Ya', 'Tidak'], 100),
})

# Label target berdasarkan logika sederhana
conditions = [
    (df['Tekanan_Darah_Sistolik'] >= 140) | (df['Tekanan_Darah_Diastolik'] >= 90),
    (df['Gula_Darah_Puasa'] >= 126)
]
choices = ['Hipertensi', 'Diabetes']
df['Diagnosis'] = np.select(conditions, choices, default='Normal')

# Label encoding
label_encoders = {}
for column in ['Jenis_Kelamin', 'Merokok', 'Aktivitas_Fisik', 'Riwayat_Keluarga_Diabetes', 'Riwayat_Keluarga_Hipertensi', 'Diagnosis']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Preprocessing
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Simpan file
joblib.dump(model, 'model_deteksi_ptm_rf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Model berhasil dilatih dan disimpan!")
