import os,pickle,numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

os.environ["ANONYMIZED_TELEMETRY"]="False"
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,"data")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")
OUT_PATH=os.path.join(DATA_DIR,"tfidf_embeddings.pkl")

def build_bm25(tfidf_matrix:csr_matrix,texts:list[str],idf:np.ndarray,k1:float=1.5,b:float=.75)->csr_matrix:
    lengths=np.array([len(t.split()) for t in texts],dtype=np.float32); avg_len=lengths.mean() if lengths.size else 1.0
    rows,cols,vals=[],[],[]; cx=tfidf_matrix.tocsr()
    for i in range(cx.shape[0]):
        start,end=cx.indptr[i],cx.indptr[i+1]; col_idx=cx.indices[start:end]; tf=cx.data[start:end].astype(np.float32)
        length_norm=1-b+b*(lengths[i]/(avg_len+1e-6)); bm_score=(tf*(k1+1))/(tf+k1*length_norm+1e-6); bm_score*=idf[col_idx]
        rows.extend([i]*len(col_idx)); cols.extend(col_idx); vals.extend(bm_score)
    return normalize(csr_matrix((vals,(rows,cols)),shape=tfidf_matrix.shape,dtype=np.float32),norm="l2")

print("Loading documents from ChromaDB...")
if not os.path.exists(CHROMA_PATH):raise RuntimeError(f"ChromaDB not found: {CHROMA_PATH}. Run pdftochunks.py first.")
embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
db=Chroma(persist_directory=CHROMA_PATH,embedding_function=embeddings); data=db.get()
texts=data.get("documents") or []; meta=data.get("metadatas") or []
if not texts:raise RuntimeError("ChromaDB is empty. Run pdftochunks.py first.")
print(f"{len(texts)} chunks loaded")
enriched=[]
for t,m in zip(texts,meta):
    m=m or {}; parts=[]
    if m.get("chapter"):parts.append(f"Chapter {m['chapter']}")
    if m.get("article"):parts.append(f"Article {m['article']}")
    if m.get("section"):parts.append(f"Section {m['section']}")
    if m.get("heading"):parts.append(m["heading"])
    if m.get("page"):parts.append(f"Page {m['page']}")
    enriched.append((" ".join(parts)+" "+(t or "")).strip())
print("Fitting TF-IDF vectorizer...")
vectorizer=TfidfVectorizer(ngram_range=(1,2),max_df=.90,min_df=2,sublinear_tf=True,strip_accents="unicode",analyzer="word")
tfidf_matrix=vectorizer.fit_transform(enriched)
print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")
print("Building BM25 matrix...")
bm25_matrix=build_bm25(tfidf_matrix,enriched,vectorizer.idf_)
out={"texts":texts,"enriched":enriched,"tfidf":bm25_matrix,"vectorizer":vectorizer,"metadatas":meta,"version":"2.0"}
with open(OUT_PATH,"wb") as f:pickle.dump(out,f,protocol=pickle.HIGHEST_PROTOCOL)
print(f"Hybrid model saved -> {OUT_PATH}")
print(f"Chunks: {len(texts)} | Vocab: {len(vectorizer.vocabulary_):,} | Matrix: {bm25_matrix.shape}")