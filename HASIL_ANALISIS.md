# Ringkasan Hasil Analisis — Telco Customer Segmentation

## Data dan preprocessing
- Observasi: 7.043 pelanggan.
- Variabel asli: 21 kolom.
- `customerID` dikeluarkan karena identifier.
- `Churn` dikeluarkan dari pembentukan klaster dan hanya dipakai untuk profiling setelah clustering.
- Fitur numerik: `tenure`, `MonthlyCharges`, `TotalCharges`.
- Fitur kategorik: 16 variabel lainnya selain ID dan target.
- `TotalCharges` dikonversi dari teks menjadi numerik; 11 nilai kosong/spasi diimputasi dengan median.
- Fitur numerik distandardisasi dengan Z-score.

## Metode
K-Prototypes (Huang-style): jumlah kuadrat jarak Euclidean pada fitur numerik terstandar ditambah gamma dikalikan jumlah ketidakcocokan kategori. Prototipe numerik adalah rata-rata, prototipe kategorik adalah modus. Pengaturan hasil baseline: k=3, gamma=0,5, 10 inisialisasi, random seed=42.

## Pemilihan k
Evaluasi dilakukan untuk k=2 sampai 6 dengan 3 inisialisasi per k. Silhouette dihitung pada sampel acak 700 observasi menggunakan jarak campuran K-Prototypes.

| k | Cost | Silhouette |
|---:|---:|---:|
| 2 | 32.864,53 | 0,2940 |
| 3 | 23.874,37 | 0,3110 |
| 4 | 21.762,61 | 0,2224 |
| 5 | 20.209,45 | 0,2148 |
| 6 | 19.260,55 | 0,1898 |

k=3 dipilih sebagai baseline karena silhouette tertinggi pada rentang yang diuji dan penurunan cost terbesar terjadi dari k=2 ke k=3; setelah itu penurunan cost relatif lebih kecil. Pemilihan akhir tetap mempertimbangkan interpretabilitas segmen.

## Profil hasil baseline (k=3, gamma=0,5)
| Klaster | Jumlah | Tenure rata-rata (bulan) | MonthlyCharges rata-rata | TotalCharges rata-rata | Churn rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.175 | 58,61 | 89,74 | 5.246,84 | 14,7% |
| 2 | 3.270 | 15,48 | 69,13 | 1.075,38 | 43,8% |
| 3 | 1.598 | 31,24 | 21,82 | 715,37 | 7,3% |

Interpretasi awal:
- **Klaster 1 — pelanggan loyal dengan nilai tagihan tinggi:** tenure rata-rata paling panjang dan total tagihan rata-rata tertinggi. Proporsi kontrak dua tahun 43,9%; churn 14,7%.
- **Klaster 2 — pelanggan baru/berjangka pendek dengan risiko churn tinggi:** 86,3% menggunakan kontrak month-to-month; 49,6% menggunakan electronic check; churn 43,8%. Segmen ini menjadi prioritas retensi.
- **Klaster 3 — pelanggan layanan dasar tanpa internet:** 95,5% tidak berlangganan internet; biaya bulanan rata-rata paling rendah; churn 7,3%.

## Catatan interpretasi
- Nomor klaster bersifat arbitrer, bukan urutan kualitas.
- Churn rate bukan variabel pembentuk klaster; angka tersebut adalah evaluasi deskriptif setelah segmentasi.
- K-Prototypes sensitif terhadap inisialisasi, gamma, dan pemilihan fitur. Dashboard memungkinkan pengaturan k, gamma, jumlah inisialisasi, dan seed.
- Implementasi K-Prototypes pada `clustering.py` mengikuti fungsi jarak dan pembaruan rata-rata/modus yang dijelaskan dalam materi kuliah. Silhouette menggunakan akar kuadrat dari dissimilarity campuran aditif sebagai matriks jarak untuk perhitungan indeks.
