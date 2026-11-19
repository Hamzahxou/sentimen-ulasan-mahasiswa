import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import preprocess

BASE = Path(__file__).parent
DATA = BASE / "data" / "ulasan_mahasiswa.csv"
MODEL_DIR, OUT_DIR = BASE / "model", BASE / "output"
LABELS = ["negatif", "netral", "positif"]
SEED = 42


def buat_model(nama: str) -> Pipeline:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    clf = (
        MultinomialNB(alpha=0.5)
        if nama == "Naive Bayes"
        else LinearSVC(C=1.0, random_state=SEED)
    )
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # 1. Muat & periksa data
    df = pd.read_csv(DATA).dropna().drop_duplicates()
    assert {"ulasan", "sentimen"} <= set(
        df.columns
    ), "CSV harus punya kolom ulasan & sentimen"
    print(f"Jumlah data : {len(df)} ulasan")
    print(df["sentimen"].value_counts().to_string(), "\n")

    # 2. Preprocessing
    df["bersih"] = df["ulasan"].apply(preprocess)
    df = df[df["bersih"].str.len() > 0]

    # 3. Bagi data latih / uji (80:20, stratified)
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        df["bersih"],
        df["sentimen"],
        df["ulasan"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["sentimen"],
    )
    print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}\n")

    # distribusi kelas
    ax = (
        df["sentimen"]
        .value_counts()
        .reindex(LABELS)
        .plot(kind="bar", color=["#d9534f", "#8a8a8a", "#3c9d5d"], rot=0)
    )
    ax.set_title("Distribusi Kelas Sentimen")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah ulasan")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "distribusi_kelas.png", dpi=150)
    plt.close()

    # 4. Latih & evaluasi tiap algoritma
    hasil, pred_tabel = {}, pd.DataFrame(
        {"ulasan": raw_test.values, "label_asli": y_test.values}
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for nama, slug in [("Naive Bayes", "naive_bayes"), ("SVM", "svm")]:
        model = buat_model(nama)
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="macro", zero_division=0
        )
        hasil[nama] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision_macro": round(p, 4),
            "recall_macro": round(r, 4),
            "f1_macro": round(f1, 4),
            "cv_f1_macro_mean": round(cv_f1.mean(), 4),
            "cv_f1_macro_std": round(cv_f1.std(), 4),
        }
        print(f"=== {nama} ===")
        print(
            f"CV 5-fold F1 macro (data latih): {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}"
        )
        print(classification_report(y_test, pred, labels=LABELS, zero_division=0))

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=LABELS, cmap="Blues"
        )
        disp.ax_.set_title(f"Confusion Matrix - {nama}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_{slug}.png", dpi=150)
        plt.close()

        pred_tabel[f"prediksi_{slug}"] = pred
        # model akhir dilatih ulang pada seluruh data agar app memakai semua data
        joblib.dump(
            buat_model(nama).fit(df["bersih"], df["sentimen"]),
            MODEL_DIR / f"{slug}.joblib",
        )

    # 5. Pilih model terbaik berdasarkan CV pada data latih (bukan data uji)
    terbaik = max(hasil, key=lambda n: hasil[n]["cv_f1_macro_mean"])
    print(f"Model terbaik (CV F1 macro): {terbaik}")

    pred_tabel.to_csv(OUT_DIR / "hasil_pengujian.csv", index=False)
    salah = pred_tabel[
        (pred_tabel.label_asli != pred_tabel.prediksi_naive_bayes)
        | (pred_tabel.label_asli != pred_tabel.prediksi_svm)
    ]
    salah.to_csv(OUT_DIR / "salah_prediksi.csv", index=False)
    print(f"Data uji yang salah diprediksi minimal oleh satu model: {len(salah)}")

    metrik = {
        "jumlah_data": len(df),
        "data_latih": len(X_train),
        "data_uji": len(X_test),
        "model_terbaik": terbaik,
        "hasil": hasil,
    }
    (OUT_DIR / "metrik.json").write_text(json.dumps(metrik, indent=2), encoding="utf-8")

    # grafik perbandingan
    pd.DataFrame(hasil).T[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ].plot(kind="bar", rot=0, figsize=(7, 4), ylim=(0, 1.05))
    plt.title("Perbandingan Kinerja Model (Data Uji)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "perbandingan_model.png", dpi=150)
    plt.close()
    print("Selesai. Model tersimpan di folder model/, hasil di folder output/")


if __name__ == "__main__":
    main()
