# Telco Customer Segmentation — K-Prototypes

Dashboard Streamlit untuk tugas analisis klaster pelanggan Telco Customer Churn.

## Isi repository
- `app.py` — dashboard interaktif Streamlit.
- `clustering.py` — implementasi K-Prototypes (komponen numerik kuadrat Euclidean + gamma simple-matching kategorik).
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — dataset tugas.
- `requirements.txt` — dependensi.

## Jalankan lokal
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Buat repository GitHub baru.
2. Upload seluruh file di folder ini ke root repository (bukan folder ZIP).
3. Buka https://share.streamlit.io/ dan login dengan GitHub.
4. Pilih **New app**, repository, branch `main`, dan file utama `app.py`.
5. Klik **Deploy**. Tunggu proses build selesai.

## Metodologi
K-Prototypes dipilih karena data memuat fitur numerik dan kategorik. `customerID` dikeluarkan karena hanya identifier. `Churn` tidak digunakan dalam pembentukan klaster (menghindari target leakage), tetapi digunakan untuk profiling segmen setelah clustering. Fitur numerik `tenure`, `MonthlyCharges`, dan `TotalCharges` distandardisasi menggunakan Z-score. `TotalCharges` dibaca sebagai numerik; 11 nilai kosong (spasi) diimputasi median.

K-Prototypes memakai jarak:
`d(x,q) = sum((x_num - q_num)^2) + gamma * jumlah_ketidakcocokan_kategorik`
Prototipe numerik diperbarui dengan rata-rata dan prototipe kategorik dengan modus. Hasil tergantung inisialisasi; dashboard menjalankan beberapa inisialisasi dan memilih cost terkecil.

Silhouette dihitung pada sampel dengan jarak campuran yang sama, setelah akar kuadrat dari dissimilarity aditif agar menjadi matriks jarak. Pertimbangkan elbow, silhouette, ukuran segmen, dan interpretabilitas secara bersama-sama.

## Catatan akademik
Implementasi `clustering.py` merupakan implementasi K-Prototypes bergaya Huang untuk pembelajaran. Untuk laporan, sebutkan formula, preprocessing, nilai k, gamma, jumlah inisialisasi, cost, dan kriteria evaluasi. Label klaster bersifat arbitrer (nomor klaster bukan ranking).
