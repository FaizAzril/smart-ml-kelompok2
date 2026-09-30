# Smart ML Kelompok 2 — Steel Plates Faults

Aplikasi Machine Learning untuk klasifikasi dan clustering cacat pada plat baja menggunakan dataset Steel Plates Faults.

## Fitur Utama

### 1. Classification

- Menggunakan model Random Forest.
- Memprediksi jenis cacat plat baja berdasarkan karakteristik input.
- Menyediakan contoh data untuk pengujian.

### 2. Clustering

- Menggunakan model K-Means.
- Mengelompokkan data berdasarkan 29 fitur karakteristik plat baja.
- Menampilkan visualisasi dua dimensi menggunakan Principal Component Analysis (PCA).
- Menampilkan jumlah data pada setiap cluster.

## Teknologi

- Python
- Flask
- pandas dan NumPy
- scikit-learn
- Matplotlib
- joblib

## Struktur Project

```text
smart-ml-kelompok2/
├── app.py
├── data/
│   ├── test.csv
│   └── steel_plates_clean.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── standard_scaler.pkl
│   ├── kmeans_model.pkl
│   ├── pca_2d.pkl
│   └── unsupervised_scaler.pkl
├── static/
│   └── css/
│       └── style.css
├── templates/
│   ├── index.html
│   └── clustering.html
├── training/
│   └── preprocessing.py
├── notebooks/
├── requirements.txt
└── README.md
```

## Instalasi dan Menjalankan Aplikasi

### 1. Clone repository

```bash
git clone https://github.com/FaizAzril/smart-ml-kelompok2.git
cd smart-ml-kelompok2
```

### 2. Buat virtual environment

```bash
python -m venv .venv
```

Aktifkan environment di Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instal dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
python app.py
```

Buka aplikasi melalui browser:

- **Classification:** http://127.0.0.1:5000/
- **Clustering:** http://127.0.0.1:5000/clustering

## Catatan

- Pastikan file dataset dan model tersedia pada folder yang sesuai.
- Model clustering menggunakan K-Means, scaler khusus clustering, dan PCA yang telah disediakan dalam folder `models/`.
- Hasil cluster merupakan pengelompokan model dan bukan label jenis cacat yang sebenarnya.
- Model tersimpan dapat bergantung pada versi scikit-learn yang digunakan saat pelatihan.

## Tim

Kelompok 2 — Machine Learning Project
