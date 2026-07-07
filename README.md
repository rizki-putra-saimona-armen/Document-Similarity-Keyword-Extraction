#  Document Similarity & Keyword Extraction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154F5B?style=for-the-badge&logo=data:image/png;base64,&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Analisis kemiripan dokumen dan ekstraksi kata kunci otomatis dari korpus berita berbahasa Indonesia menggunakan TF-IDF**

</div>

---

##  Deskripsi

Proyek ini mengimplementasikan dua teknik dasar dalam *Natural Language Processing* (NLP) menggunakan pendekatan **TF-IDF (Term Frequency–Inverse Document Frequency)**:

1. **Document Similarity** — mengukur seberapa mirip suatu dokumen berita dengan dokumen lain dalam korpus menggunakan *cosine similarity*.
2. **Keyword Extraction** — mengekstrak kata/frasa paling representatif (khas) dari sebuah dokumen berdasarkan bobot TF-IDF-nya.

Dataset yang digunakan adalah kumpulan artikel berita berbahasa Indonesia dari **Kompas** (2.008 dokumen).

---

##  Fitur Utama

-  **Pencarian dokumen paling mirip** menggunakan cosine similarity terhadap seluruh korpus
-  **Ekstraksi keyword otomatis** dari teks baru (di luar korpus) menggunakan vocabulary TF-IDF yang sudah dilatih
- 🇮🇩 Preprocessing khusus teks Bahasa Indonesia (stopwords + tokenisasi NLTK)
-  Analisis kata paling langka (*rarest terms*) berdasarkan skor IDF
- Dukungan **unigram & bigram** (`ngram_range=(1,2)`) untuk menangkap frasa, bukan hanya kata tunggal

---

##  Tech Stack

| Komponen | Library |
|---|---|
| Data handling | `pandas` |
| Tokenisasi & stopwords | `nltk` |
| Vektorisasi teks | `scikit-learn` (`TfidfVectorizer`) |
| Perhitungan kemiripan | `scikit-learn` (`cosine_similarity`) |

---

##  Struktur Proyek

```
 document-similarity-keyword-extraction
├── 📓 Document_Similarity_dan_Keyword_extraction.ipynb
├── 📁 data/
│   └── kompas.csv
└── 📄 README.md
```

---

##  Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/document-similarity-keyword-extraction.git
cd document-similarity-keyword-extraction
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn nltk
```

### 3. Download resource NLTK yang dibutuhkan
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Jalankan notebook
```bash
jupyter notebook "Document_Similarity_dan_Keyword_extraction.ipynb"
```

>  Pastikan file `data/kompas.csv` tersedia di direktori `data/` sebelum menjalankan notebook.

---

##  Alur Kerja (Workflow)

```
1. Load data berita (kompas.csv)
        ↓
2. Preprocessing (tokenisasi + stopword removal Bahasa Indonesia)
        ↓
3. Vektorisasi teks dengan TF-IDF (unigram + bigram)
        ↓
4. Document Similarity → cosine_similarity antar dokumen
        ↓
5. Keyword Extraction → ambil kata dengan skor TF-IDF tertinggi
```

---

##  Contoh Penggunaan

### Mencari dokumen paling mirip
```python
sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
sim.argsort()  # index diurutkan dari paling tidak mirip -> paling mirip
```

### Ekstraksi keyword dari teks baru
```python
def extract_keywords_tfidf(doc, tfidf, topk=10):
    matrix = tfidf.transform([doc])
    vocab = tfidf.get_feature_names_out()
    sorted_tfidf = matrix[0].toarray()[0].argsort()
    return [vocab[idx] for idx in reversed(sorted_tfidf[-topk:])]

extract_keywords_tfidf(text, tfidf)
# Output: ['gempa', 'bmkg', 'magnitudo', 'sukabumi', ...]
```

---

##  Contoh Hasil

**Dokumen uji:** Berita kasus Ginandjar Kartasasmita
**Dokumen paling mirip ditemukan:** Berita lain yang juga membahas Ginandjar Kartasasmita & Kejaksaan Agung ✅

Ini membuktikan bahwa pendekatan TF-IDF + cosine similarity berhasil menangkap **kesamaan topik** antar dokumen berita, bukan sekadar kesamaan kata acak.

---

##  Konsep yang Dipelajari

- TF-IDF (Term Frequency–Inverse Document Frequency)
- Cosine Similarity untuk pengukuran kemiripan teks
- N-gram (unigram & bigram) dalam representasi teks
- Sparse matrix vs dense array (`toarray()`)
- Ekstraksi keyword tanpa supervised learning (*unsupervised keyword extraction*)

---

##  Bagian dari Seri

Notebook ini merupakan **Part 4** dari rangkaian pembelajaran *Text Mining / NLP* berbahasa Indonesia.

---

##  Lisensi

Proyek ini dibuat untuk tujuan pembelajaran (akademik).

---
