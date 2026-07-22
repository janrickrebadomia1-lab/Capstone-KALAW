import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ─────────────────────────────────────────────────────────
# PATHS  (must match app.py and pdftochunks.py)
# ─────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
OUT_PATH    = os.path.join(DATA_DIR, "tfidf_embeddings.pkl")

# ─────────────────────────────────────────────────────────
# BM25 SCORING  (sparse-safe — no .toarray() on full matrix)
# ─────────────────────────────────────────────────────────

def build_bm25(tfidf_matrix: csr_matrix, texts: list[str], idf: np.ndarray,
               k1: float = 1.5, b: float = 0.75) -> csr_matrix:
    """
    Applies BM25 term-weighting to a TF-IDF sparse matrix.
    Returns a normalised sparse matrix — same shape as input.
    Avoids .toarray() so it works on large corpora without OOM.
    """
    lengths = np.array([len(t.split()) for t in texts], dtype=np.float32)
    avg_len = lengths.mean() if lengths.size else 1.0

    # Work row-by-row to stay sparse
    rows, cols, vals = [], [], []

    cx = tfidf_matrix.tocsr()
    for i in range(cx.shape[0]):
        start, end = cx.indptr[i], cx.indptr[i + 1]
        col_idx = cx.indices[start:end]
        tf      = cx.data[start:end].astype(np.float32)

        length_norm = 1 - b + b * (lengths[i] / (avg_len + 1e-6))
        bm_score    = (tf * (k1 + 1)) / (tf + k1 * length_norm + 1e-6)
        bm_score   *= idf[col_idx]

        rows.extend([i] * len(col_idx))
        cols.extend(col_idx)
        vals.extend(bm_score)

    bm_sparse = csr_matrix(
        (vals, (rows, cols)),
        shape=tfidf_matrix.shape,
        dtype=np.float32,
    )

    # L2-normalise rows so dot-product with query vector gives cosine scores
    return normalize(bm_sparse, norm="l2")

# ─────────────────────────────────────────────────────────
# LOAD CHROMA DOCUMENTS
# ─────────────────────────────────────────────────────────

print("📂 Loading documents from ChromaDB…")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
db   = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
data = db.get()

texts = data["documents"]
meta  = data["metadatas"]

if not texts:
    raise RuntimeError("ChromaDB is empty — run pdftochunks.py first.")

print(f"{len(texts)} chunks loaded")

# ─────────────────────────────────────────────────────────
# BUILD ENRICHED TEXT  (metadata prefix → better keyword match)
# Now includes 'page' since pdftochunks.py stores it
# ─────────────────────────────────────────────────────────

enriched = []
for t, m in zip(texts, meta):
    parts = []
    if m.get("chapter"): parts.append(f"Chapter {m['chapter']}")
    if m.get("article"): parts.append(f"Article {m['article']}")
    if m.get("section"): parts.append(f"Section {m['section']}")
    if m.get("heading"): parts.append(m["heading"])
    if m.get("page"):    parts.append(f"Page {m['page']}")

    prefix = " ".join(parts)
    enriched.append((prefix + " " + t).strip())

# ─────────────────────────────────────────────────────────
# TF-IDF VECTORISER
# Changes from old version:
#   ngram_range (1,3) → (1,2)  — trigrams add noise + slow query time
#   min_df      1     → 2      — drop hapax legomena (garbage tokens)
# ─────────────────────────────────────────────────────────

print("⚙️  Fitting TF-IDF vectoriser…")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),    # unigrams + bigrams only
    max_df=0.90,           # ignore terms in > 90 % of docs (stopword-like)
    min_df=2,              # ignore terms appearing in only 1 doc
    sublinear_tf=True,     # log(tf) smoothing
    strip_accents="unicode",
    analyzer="word",
)

tfidf_matrix = vectorizer.fit_transform(enriched)
print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

# ─────────────────────────────────────────────────────────
# BM25 MATRIX  (replaces plain TF-IDF as the search matrix)
# Stored under key "tfidf" so app.py works without any changes
# ─────────────────────────────────────────────────────────

print("⚙️  Building BM25 matrix…")
bm25_matrix = build_bm25(tfidf_matrix, enriched, vectorizer.idf_)

# ─────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────

out = {
    "texts":      texts,       # raw chunk text (what app.py returns to LLM)
    "enriched":   enriched,    # metadata-prefixed text (what vectoriser was trained on)
    "tfidf":      bm25_matrix, # ← BM25 matrix stored as "tfidf" so app.py reads it automatically
    "vectorizer": vectorizer,
    "metadatas":  meta,
}

with open(OUT_PATH, "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)  # faster + smaller file

print(f"✅ Hybrid model saved → {OUT_PATH}")
print(f"   Chunks: {len(texts)} | Vocab: {len(vectorizer.vocabulary_):,} | Matrix: {bm25_matrix.shape}")