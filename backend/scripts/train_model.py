import os,pickle,math,re
from collections import Counter,defaultdict
from langchain_chroma import Chroma

os.environ["ANONYMIZED_TELEMETRY"]="False"

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR,"..","data")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")
OUT_PATH=os.path.join(DATA_DIR,"tfidf_embeddings.pkl")

def tokenize(text:str)->list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25_index(texts:list[str], k1:float=1.5, b:float=0.75)->dict:
    n=len(texts)
    lengths=[]
    postings=defaultdict(dict)
    df=Counter()
    for doc_id,text in enumerate(texts):
        counts=Counter(tokenize(text))
        lengths.append(sum(counts.values()))
        for term,tf in counts.items():
            postings[term][doc_id]=tf
            df[term]+=1
    avg_len=(sum(lengths)/n) if n else 1.0
    idf={term:math.log(1.0+(n-f+0.5)/(f+0.5)) for term,f in df.items()}
    return {"postings":dict(postings),"doc_lengths":lengths,"idf":idf,"avg_doc_len":avg_len,"k1":k1,"b":b,"document_count":n}

print("Loading documents from ChromaDB...")

if not os.path.exists(CHROMA_PATH):
    raise RuntimeError(
        f"ChromaDB not found: {CHROMA_PATH}. "
        "Run pdf_to_chunks.py first."
    )

db=Chroma(persist_directory=CHROMA_PATH)

data=db.get()

texts=data.get("documents") or []
meta=data.get("metadatas") or []

if not texts:
    raise RuntimeError(
        "ChromaDB is empty. Run pdf_to_chunks.py first."
    )

print(f"{len(texts)} chunks loaded.")

enriched=[]

for text,metadata in zip(texts,meta):
    metadata=metadata or {}
    parts=[]

    if metadata.get("chapter"):
        parts.append(f"Chapter {metadata['chapter']}")

    if metadata.get("article"):
        parts.append(f"Article {metadata['article']}")

    if metadata.get("section"):
        parts.append(f"Section {metadata['section']}")

    if metadata.get("heading"):
        parts.append(metadata["heading"])

    if metadata.get("page"):
        parts.append(f"Page {metadata['page']}")

    prefix=" ".join(parts)
    enriched.append(
        (prefix+" "+(text or "")).strip()
    )

print("Building true BM25 index...")
bm25=build_bm25_index(enriched)
out={"texts":texts,"enriched":enriched,"metadatas":meta,"bm25":bm25,"version":"4.0-bm25"}

with open(OUT_PATH,"wb") as f:
    pickle.dump(
        out,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(f"Hybrid model saved -> {OUT_PATH}")
print(f"Chunks: {len(texts)} | Terms: {len(bm25["idf"]):,}")
