# Analisis Sentimen Ulasan Mahasiswa

Proyek Akhir Kecerdasan Buatan — Tema 3. Mengklasifikasikan ulasan mahasiswa menjadi
**positif, netral, negatif** dengan TF-IDF + **Naive Bayes** (utama) dan **SVM** (pembanding).

## Cara menjalankan
```bash
pip install -r requirements.txt
python train.py            # latih + evaluasi, hasil di folder model/ dan output/
python predict.py "Dosennya menjelaskan dengan jelas, saya puas"   # uji lewat terminal
streamlit run app.py       # aplikasi demo (browser)
```

## Struktur
| File | Fungsi |
|---|---|
| `data/ulasan_mahasiswa.csv` | Dataset (kolom `ulasan`, `sentimen`) |
| `preprocessing.py` | Case folding, cleaning, normalisasi kata gaul, tokenisasi, stopword removal |
| `train.py` | Split 80:20 (stratified), TF-IDF, latih NB & SVM, cross-validation, evaluasi, grafik |
| `predict.py` | Fungsi prediksi + CLI + kata yang paling memengaruhi prediksi |
| `app.py` | Aplikasi Streamlit: input -> preprocessing -> prediksi -> output |
| `output/` | `metrik.json`, `hasil_pengujian.csv`, `salah_prediksi.csv`, grafik PNG untuk PPT |

## Alur sistem
Ulasan -> preprocessing -> TF-IDF (unigram + bigram) -> klasifikasi (Naive Bayes / SVM) -> label + probabilitas.

Kata negasi (`tidak`, `kurang`, `belum`, `bukan`) sengaja tidak dihapus sebagai stopword, dan bigram
membuat frasa seperti "tidak bagus" tetap terbaca sebagai satu fitur.

## Catatan dataset
Dataset berisi 348 ulasan (116 per kelas) yang **ditulis sendiri oleh tim sebagai data contoh** — bukan
hasil scraping atau survei asli. Untuk hasil yang lebih kuat, tambahkan ulasan nyata (mis. Google Form ke
teman satu angkatan) ke file CSV yang sama dengan format kolom yang sama, lalu jalankan ulang `python train.py`.

## Keterbatasan
- Data uji hanya 70 ulasan, sehingga selisih kecil antara Naive Bayes dan SVM belum bisa dianggap bermakna.
- Belum ada stemming Bahasa Indonesia; sarkasme dan ulasan campuran (puji + keluhan) masih sulit.
