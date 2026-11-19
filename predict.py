import argparse
from pathlib import Path

import joblib
import numpy as np

from preprocessing import preprocess_steps

MODEL_DIR = Path(__file__).parent / "model"
NAMA_MODEL = {"naive_bayes": "Naive Bayes", "svm": "SVM (Linear)"}
_cache = {}


def muat_model(slug: str = "naive_bayes"):
    if slug not in _cache:
        path = MODEL_DIR / f"{slug}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model {path.name} belum ada. Jalankan dulu: python train.py"
            )
        _cache[slug] = joblib.load(path)
    return _cache[slug]


def _skor_kelas(pipe, bersih: str) -> dict:
    """Probabilitas (Naive Bayes) atau skor relatif softmax (SVM) untuk tiap kelas."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        nilai = pipe.predict_proba([bersih])[0]
    else:
        d = pipe.decision_function([bersih])[0]
        e = np.exp(d - d.max())
        nilai = e / e.sum()
    return {k: float(v) for k, v in zip(clf.classes_, nilai)}


def kata_berpengaruh(pipe, bersih: str, label: str, top: int = 5) -> list:
    """Kata-kata (fitur TF-IDF) yang paling mendorong prediksi ke kelas `label`."""
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    x = tfidf.transform([bersih])
    kelas = list(clf.classes_)
    i = kelas.index(label)
    if hasattr(clf, "feature_log_prob_"):  # Naive Bayes: selisih log-prob vs kelas lain
        lain = [j for j in range(len(kelas)) if j != i]
        bobot = clf.feature_log_prob_[i] - clf.feature_log_prob_[lain].mean(axis=0)
    else:  # SVM linear: koefisien kelas
        bobot = clf.coef_[i]
    idx = x.nonzero()[1]
    nama = tfidf.get_feature_names_out()
    hasil = sorted(
        ((nama[j], float(x[0, j] * bobot[j])) for j in idx), key=lambda t: -t[1]
    )
    return [(k, round(s, 4)) for k, s in hasil[:top] if s > 0]


def prediksi(teks: str, slug: str = "naive_bayes") -> dict:
    pipe = muat_model(slug)
    langkah = preprocess_steps(teks)
    bersih = langkah["teks_bersih"]
    if not bersih:
        return {"label": None, "skor": {}, "langkah": langkah, "kata": []}
    skor = _skor_kelas(pipe, bersih)
    label = max(skor, key=skor.get)
    return {
        "label": label,
        "skor": skor,
        "langkah": langkah,
        "kata": kata_berpengaruh(pipe, bersih, label),
    }


def main():
    ap = argparse.ArgumentParser(description="Prediksi sentimen ulasan mahasiswa")
    ap.add_argument("teks", nargs="?", help="ulasan yang akan dianalisis")
    ap.add_argument("--model", choices=list(NAMA_MODEL), default="naive_bayes")
    args = ap.parse_args()

    def tampil(t):
        h = prediksi(t, args.model)
        if h["label"] is None:
            print("Teks kosong / tidak mengandung kata yang dikenali.\n")
            return
        print(f"Teks bersih : {h['langkah']['teks_bersih']}")
        print(f"Sentimen    : {h['label'].upper()}  ({NAMA_MODEL[args.model]})")
        print(
            "Skor        : " + ", ".join(f"{k}={v:.1%}" for k, v in h["skor"].items())
        )
        if h["kata"]:
            print("Kata kunci  : " + ", ".join(k for k, _ in h["kata"]))
        print()

    if args.teks:
        tampil(args.teks)
    else:
        print(f"Mode interaktif ({NAMA_MODEL[args.model]}). Ketik 'q' untuk keluar.\n")
        while True:
            t = input("Ulasan> ").strip()
            if t.lower() in {"q", "quit", "keluar"}:
                break
            if t:
                tampil(t)


if __name__ == "__main__":
    main()
