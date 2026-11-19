
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import NAMA_MODEL, muat_model, prediksi
from preprocessing import preprocess

BASE = Path(__file__).parent
OUT = BASE / "output"
WARNA = {"positif": "Positif", "netral": "Netral", "negatif": "Negatif"}

st.set_page_config(page_title="Analisis Sentimen Ulasan Mahasiswa", page_icon="💬", layout="wide")


def model_siap() -> bool:
    return all((BASE / "model" / f"{s}.joblib").exists() for s in NAMA_MODEL)


def tampil_hasil(label: str):
    teks = f"Sentimen: **{WARNA[label]}**"
    {"positif": st.success, "negatif": st.error, "netral": st.info}[label](teks)


st.title("Analisis Sentimen Ulasan Mahasiswa")
st.caption("Klasifikasi ulasan menjadi positif, netral, atau negatif — TF-IDF + Naive Bayes / SVM")

if not model_siap():
    st.error("Model belum dilatih. Jalankan `python train.py` di terminal, lalu muat ulang halaman ini.")
    st.stop()

with st.sidebar:
    st.header("Pengaturan")
    slug = st.radio("Algoritma", list(NAMA_MODEL), format_func=lambda s: NAMA_MODEL[s])
    st.markdown("**Alur sistem**\n\n1. Input ulasan\n2. Preprocessing\n3. TF-IDF\n4. Klasifikasi\n5. Output sentimen")

tab1, tab2, tab3 = st.tabs(["Analisis Ulasan", "Analisis File CSV", "Performa Model"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    contoh = {
        "(pilih contoh)": "",
        "Positif": "Dosennya menjelaskan dengan jelas dan sabar, saya jadi paham materinya",
        "Negatif": "Wifi kampus lemot banget, buka e-learning aja susah dan bikin kesal",
        "Netral": "Praktikum basis data diadakan setiap hari Rabu di lab komputer",
    }
    pilih = st.selectbox("Coba contoh ulasan (opsional)", list(contoh))
    teks = st.text_area("Tulis ulasan mahasiswa", value=contoh[pilih], height=110,
                        placeholder="Contoh: Pelayanan akademik cepat dan stafnya ramah")

    if st.button("Analisis Sentimen", type="primary"):
        if not teks.strip():
            st.warning("Masukkan ulasan terlebih dahulu.")
        else:
            h = prediksi(teks, slug)
            if h["label"] is None:
                st.warning("Ulasan tidak mengandung kata yang dapat dianalisis (hanya angka/simbol?).")
            else:
                kiri, kanan = st.columns(2)
                with kiri:
                    st.subheader("Output")
                    tampil_hasil(h["label"])
                    label_skor = "Probabilitas" if slug == "naive_bayes" else "Skor relatif (softmax)"
                    st.write(f"**{label_skor} tiap kelas**")
                    st.bar_chart(pd.Series(h["skor"], name="skor"))
                    if h["kata"]:
                        st.write("**Kata yang paling memengaruhi prediksi**")
                        st.dataframe(pd.DataFrame(h["kata"], columns=["kata / frasa", "pengaruh"]),
                                     hide_index=True)
                with kanan:
                    st.subheader("Tahap preprocessing")
                    L = h["langkah"]
                    st.markdown("**1. Teks asli**");            st.code(L["teks_asli"], language=None)
                    st.markdown("**2. Case folding**");         st.code(L["case_folding"], language=None)
                    st.markdown("**3. Cleaning**");             st.code(L["cleaning"], language=None)
                    st.markdown("**4. Normalisasi kata gaul**"); st.code(" ".join(L["normalisasi"]), language=None)
                    st.markdown("**5. Stopword removal**");     st.code(L["teks_bersih"], language=None)

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.write("Unggah file CSV dengan kolom **ulasan** untuk menganalisis banyak ulasan sekaligus.")
    file = st.file_uploader("File CSV", type="csv")
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"File tidak dapat dibaca: {e}")
            df = None
        if df is not None:
            if "ulasan" not in df.columns:
                st.error("Kolom `ulasan` tidak ditemukan. Pastikan nama kolomnya persis 'ulasan'.")
            else:
                df = df.dropna(subset=["ulasan"]).reset_index(drop=True)
                pipe = muat_model(slug)
                bersih = df["ulasan"].astype(str).apply(preprocess)
                ok = bersih.str.len() > 0
                df["sentimen_prediksi"] = None
                df.loc[ok, "sentimen_prediksi"] = pipe.predict(bersih[ok])
                st.write(f"{int(ok.sum())} dari {len(df)} ulasan berhasil dianalisis.")
                st.bar_chart(df["sentimen_prediksi"].value_counts())
                st.dataframe(df, hide_index=True)
                st.download_button("Unduh hasil (CSV)", df.to_csv(index=False).encode("utf-8"),
                                   "hasil_sentimen.csv", "text/csv")

# ---------------------------------------------------------------- Tab 3
with tab3:
    metrik_path = OUT / "metrik.json"
    if not metrik_path.exists():
        st.info("Belum ada hasil evaluasi. Jalankan `python train.py`.")
    else:
        m = json.loads(metrik_path.read_text(encoding="utf-8"))
        c = st.columns(4)
        c[0].metric("Jumlah data", m["jumlah_data"])
        c[1].metric("Data latih", m["data_latih"])
        c[2].metric("Data uji", m["data_uji"])
        c[3].metric("Model terbaik (CV)", m["model_terbaik"])
        tabel = pd.DataFrame(m["hasil"]).T.rename(columns={
            "accuracy": "Accuracy", "precision_macro": "Precision", "recall_macro": "Recall",
            "f1_macro": "F1-Score", "cv_f1_macro_mean": "CV F1 (rata-rata)", "cv_f1_macro_std": "CV F1 (std)"})
        st.dataframe(tabel)
        gambar = [("distribusi_kelas.png", "Distribusi kelas"),
                  ("perbandingan_model.png", "Perbandingan model"),
                  ("confusion_matrix_naive_bayes.png", "Confusion matrix Naive Bayes"),
                  ("confusion_matrix_svm.png", "Confusion matrix SVM")]
        kolom = st.columns(2)
        for i, (f, cap) in enumerate(gambar):
            if (OUT / f).exists():
                with kolom[i % 2]:
                    st.image(str(OUT / f), caption=cap)
