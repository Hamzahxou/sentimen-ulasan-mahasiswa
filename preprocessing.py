import re

KAMUS_NORMALISASI = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "bgt": "banget",
    "bngt": "banget",
    "yg": "yang",
    "dgn": "dengan",
    "krn": "karena",
    "karna": "karena",
    "sy": "saya",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "jg": "juga",
    "utk": "untuk",
    "dr": "dari",
    "tp": "tapi",
    "bgus": "bagus",
    "mantul": "mantap",
    "mantab": "mantap",
    "bosen": "bosan",
    "aja": "saja",
    "gimana": "bagaimana",
    "kmrn": "kemarin",
    "lg": "lagi",
    "dll": "dan lain lain",
    "sm": "sama",
    "org": "orang",
    "nyesel": "menyesal",
    "kepake": "terpakai",
    "ngerti": "mengerti",
    "nyangka": "menyangka",
    "dosennya": "dosen",
    "materinya": "materi",
}

STOPWORDS = {
    "yang",
    "di",
    "ke",
    "dari",
    "dan",
    "itu",
    "ini",
    "untuk",
    "pada",
    "dengan",
    "saya",
    "aku",
    "kami",
    "kita",
    "adalah",
    "juga",
    "atau",
    "oleh",
    "dalam",
    "akan",
    "para",
    "sebuah",
    "sih",
    "nih",
    "deh",
    "lah",
    "kok",
    "ya",
    "pak",
    "bu",
}


def case_folding(teks: str) -> str:
    return str(teks).lower()


def cleaning(teks: str) -> str:
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)  # URL
    teks = re.sub(r"[@#]\w+", " ", teks)  # mention / hashtag
    teks = re.sub(r"[^a-z\s]", " ", teks)  # angka, tanda baca, emoji
    teks = re.sub(r"(.)\1{2,}", r"\1", teks)  # 'bagusss' -> 'bagus'
    return re.sub(r"\s+", " ", teks).strip()


def normalisasi(tokens: list) -> list:
    hasil = []
    for t in tokens:
        hasil.extend(KAMUS_NORMALISASI.get(t, t).split())
    return hasil


def hapus_stopword(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def preprocess_steps(teks: str) -> dict:
    """Mengembalikan hasil setiap tahap (dipakai untuk demonstrasi di aplikasi)."""
    lower = case_folding(teks)
    bersih = cleaning(lower)
    token = bersih.split()
    normal = normalisasi(token)
    final = hapus_stopword(normal)
    return {
        "teks_asli": teks,
        "case_folding": lower,
        "cleaning": bersih,
        "tokenisasi": token,
        "normalisasi": normal,
        "stopword_removal": final,
        "teks_bersih": " ".join(final),
    }


def preprocess(teks: str) -> str:
    return preprocess_steps(teks)["teks_bersih"]
