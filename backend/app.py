import os
import json
import asyncio
import logging
import pickle
import hashlib
import re
import random
from difflib import SequenceMatcher
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
os.environ["ANONYMIZED_TELEMETRY"] = "False"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("kalaw")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
HYBRID_PATH = os.path.join(DATA_DIR, "tfidf_embeddings.pkl")
JSON_PATH   = os.path.join(DATA_DIR, "faculty_manual.json")

# Tuning knobs
JSON_SCORE_THRESHOLD   = 0.65   # min similarity for JSON match
CHROMA_TOP_K           = 5      # final semantic chunks returned
CHROMA_FETCH_K         = 12     # candidate pool for MMR
KEYWORD_TOP_K          = 4      # TF-IDF chunks
MAX_CONTEXT_CHUNKS     = 4      # chunks sent to LLM
MAX_CHUNK_CHARS        = 1000   # truncate each chunk
CACHE_MAX              = 150    # response cache size
HISTORY_WINDOW         = 4      # last N messages kept per session

# ─────────────────────────────────────────────────────────
# LOAD JSON INTENTS  (pre-index for fast lookup)
# ─────────────────────────────────────────────────────────

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        INTENTS = json.load(f)
else:
    INTENTS = []

# Pre-tokenise every pattern once at startup
_INTENT_INDEX: list[dict] = []
for intent in INTENTS:
    for pattern in intent.get("patterns", []):
        tokens = set(re.findall(r"[a-z0-9]+", pattern.lower()))
        _INTENT_INDEX.append({
            "pattern":  pattern.lower(),
            "tokens":   tokens,
            "response": intent.get("response"),
        })

log.info("JSON intents loaded: %d patterns", len(_INTENT_INDEX))

_GREETING_TRIGGERS = {
    # English
    "hi", "hello", "hey", "goodmorning", "goodafternoon", "goodevening",
    "good day", "howdy", "greetings", "hi there", "hello there", "hey there",
    "what's up", "sup", "yo",
    # Filipino / Tagalog
    "kamusta", "kumusta", "musta", "magandang umaga", "magandang hapon",
    "magandang gabi", "magandang araw",
    # Bisaya / Cebuano
    "maayong buntag", "maayong udto", "maayong hapon", "maayong gabii",
    "kumusta ka",
}

_GREETING_RESPONSES = [
    (
        "Hello! I'm **KALAW**, your CPSU Faculty Manual assistant.\n\n"
        "I can help you with policies, procedures, leave benefits, faculty ranks, "
        "and anything covered in the Faculty Manual. What would you like to know?"
    ),
    (
        "Hi there! Welcome — I'm **KALAW**, the CPSU Faculty Manual chatbot.\n\n"
        "Feel free to ask me about faculty policies, duties, benefits, or any section "
        "of the manual. How can I assist you today?"
    ),
    (
        "Good day! I'm **KALAW**, here to help you navigate the CPSU Faculty Manual.\n\n"
        "Ask me anything about faculty rules, leave entitlements, promotions, or "
        "academic procedures. What's your question?"
    ),
    (
        "Hey! I'm **KALAW** — your go-to guide for the CPSU Faculty Manual.\n\n"
        "Whether it's about teaching loads, leave policies, or faculty obligations, "
        "I'm ready to help. What do you need?"
    ),
]


def greeting_match(query: str) -> str | None:
    """
    Returns a randomised greeting response if the query is a greeting,
    otherwise None. Fires before the JSON / RAG layers — zero DB cost.
    """
    q = re.sub(r"[^a-z0-9\s]", " ", query.lower()).strip()
    q = re.sub(r"\s+", " ", q)

    # Exact match
    if q in _GREETING_TRIGGERS:
        return random.choice(_GREETING_RESPONSES)

    # Starts-with match: "hello po", "hi kalaw", "good morning sir", etc.
    for trigger in _GREETING_TRIGGERS:
        if q.startswith(trigger + " ") or q == trigger:
            return random.choice(_GREETING_RESPONSES)

    return None

# ─────────────────────────────────────────────────────────
# JSON INTENT MATCH
# ─────────────────────────────────────────────────────────

def json_match(query: str) -> str | None:
    """
    Two-pass lookup:
      1. Token-overlap ratio  → fast O(n) pre-filter
      2. SequenceMatcher      → precise score on survivors
    Returns the best response or None.
    """
    q_norm   = query.lower().strip()
    q_tokens = set(re.findall(r"[a-z0-9]+", q_norm))

    best_score    = 0.0
    best_response = None

    for entry in _INTENT_INDEX:
        # Fast token overlap pre-filter
        if q_tokens and entry["tokens"]:
            overlap = len(q_tokens & entry["tokens"]) / max(len(q_tokens), len(entry["tokens"]))
            if overlap < 0.20:
                continue

        score = SequenceMatcher(None, q_norm, entry["pattern"]).ratio()

        # Bonus for substring containment
        if entry["pattern"] in q_norm or q_norm in entry["pattern"]:
            score += 0.25

        if score > best_score and score >= JSON_SCORE_THRESHOLD:
            best_score    = score
            best_response = entry["response"]

    return best_response

# ─────────────────────────────────────────────────────────
# QUERY NORMALIZATION
# ─────────────────────────────────────────────────────────

def normalize_query(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

# ─────────────────────────────────────────────────────────
# MODELS  (loaded once at import time)
# ─────────────────────────────────────────────────────────

embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

llm = OllamaLLM(
    model="llama3.2:latest",
    temperature=0,
    num_ctx=2048,
    num_predict=350,
    repeat_penalty=1.1,
    top_k=20,
    top_p=0.85,
)

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
)

hybrid_data: dict | None = None
if os.path.exists(HYBRID_PATH):
    with open(HYBRID_PATH, "rb") as f:
        hybrid_data = pickle.load(f)
    log.info("TF-IDF index loaded: %d documents", len(hybrid_data["texts"]))

# ─────────────────────────────────────────────────────────
# CACHE & SESSION MEMORY
# ─────────────────────────────────────────────────────────

session_store:   dict[str, list] = {}
_response_cache: dict[str, str]  = {}
_embed_cache:    dict[str, list[float]] = {}


def cache_key(q: str) -> str:
    return hashlib.md5(normalize_query(q).encode()).hexdigest()


def get_or_embed(query: str) -> list[float]:
    """Return cached embedding or compute + cache a new one."""
    ck = cache_key(query)
    if ck not in _embed_cache:
        _embed_cache[ck] = embeddings.embed_query(query)
        if len(_embed_cache) > 500:
            _embed_cache.pop(next(iter(_embed_cache)))
    return _embed_cache[ck]

# ─────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────

PROMPT = PromptTemplate(
    input_variables=["context", "history", "question"],
    template="""You are KALAW, a strict CPSU Faculty Manual assistant.

RULES:
- Use ONLY the provided context.
- Do NOT guess or hallucinate.
- If the answer is not in the context, say exactly: "Not found in the Faculty Manual."
- Be precise, concise, and cite article/section numbers when available.

CONTEXT:
{context}

HISTORY:
{history}

QUESTION:
{question}

ANSWER:"""
)

# ─────────────────────────────────────────────────────────
# HYBRID SEARCH  (TF-IDF keyword layer)
# ─────────────────────────────────────────────────────────

def keyword_search(query: str, top_k: int = KEYWORD_TOP_K) -> list[dict]:
    if not hybrid_data:
        return []

    vec    = hybrid_data["vectorizer"].transform([query])
    scores = (hybrid_data["tfidf"] * vec.T).toarray().flatten()
    top_idx = scores.argsort()[-top_k:][::-1]

    return [
        {
            "text":     hybrid_data["texts"][i],
            "metadata": hybrid_data["metadatas"][i],
            "score":    float(scores[i]),
        }
        for i in top_idx
        if scores[i] > 0.01
    ]

# ─────────────────────────────────────────────────────────
# SEMANTIC SEARCH  (ChromaDB, uses cached embedding)
# ─────────────────────────────────────────────────────────

def semantic_search(query: str) -> list:
    """MMR re-ranking for diversity. Falls back to plain similarity search."""
    try:
        return vector_db.max_marginal_relevance_search_by_vector(
            get_or_embed(query),
            k=CHROMA_TOP_K,
            fetch_k=CHROMA_FETCH_K,
            lambda_mult=0.6,
        )
    except Exception:
        return vector_db.similarity_search(query, k=CHROMA_TOP_K)

# ─────────────────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────

def fuse(semantic_docs: list, keyword_docs: list[dict], k: int = 60) -> list[dict]:
    """Combines semantic and keyword results without score normalisation."""
    scores: dict[str, float] = {}
    store:  dict[str, dict]  = {}

    for i, d in enumerate(semantic_docs):
        key = d.page_content[:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
        store[key]  = {"text": d.page_content, "metadata": d.metadata}

    for i, d in enumerate(keyword_docs):
        key = d["text"][:120]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + i + 1)
        if key not in store:
            store[key] = d

    return sorted(
        store.values(),
        key=lambda x: scores.get(x["text"][:120], 0.0),
        reverse=True,
    )

# ─────────────────────────────────────────────────────────
# CONTEXT CLEANING
# ─────────────────────────────────────────────────────────

_BAD_KEYWORDS = {"table of contents", "index", "copyright", "acknowledgement"}


def format_context(docs: list[dict], max_chunks: int = MAX_CONTEXT_CHUNKS) -> str:
    seen: set[str] = set()
    out:  list[str] = []

    for d in docs:
        text = d["text"].strip()

        if len(text) < 100:
            continue

        text_lower = text.lower()
        if any(bad in text_lower for bad in _BAD_KEYWORDS):
            continue

        key = text[:120]
        if key in seen:
            continue
        seen.add(key)

        out.append(text[:MAX_CHUNK_CHARS])

        if len(out) >= max_chunks:
            break

    return "\n\n---\n\n".join(out)

# ─────────────────────────────────────────────────────────
# CHAT ENDPOINT
# ─────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    data       = await request.json()
    question   = data.get("question", "").strip()
    session_id = data.get("session_id", "default")

    if not question:
        return {"error": "Empty question"}

    # ── Session memory ───────────────────────────────────
    if session_id not in session_store:
        session_store[session_id] = []
    history = session_store[session_id]

    # ── Layer 0: Greeting fast path ──────────────────────
    greeting = greeting_match(question)
    if greeting:
        log.info("Greeting hit for: %s", question[:60])

        async def greeting_stream():
            yield f"data: {json.dumps({'content': greeting, 'source': 'greeting'})}\n\n"

        return StreamingResponse(greeting_stream(), media_type="text/event-stream")

    # ── Layer 1: JSON fast lookup ────────────────────────
    json_answer = json_match(question)
    if json_answer:
        log.info("JSON hit for: %s", question[:60])

        async def fast_stream():
            yield f"data: {json.dumps({'content': json_answer, 'source': 'json'})}\n\n"

        return StreamingResponse(fast_stream(), media_type="text/event-stream")

    # ── Layer 2a: Response cache ─────────────────────────
    ck = cache_key(question)
    if ck in _response_cache:
        log.info("Cache hit for: %s", question[:60])
        cached = _response_cache[ck]

        async def cached_stream():
            yield f"data: {json.dumps({'content': cached, 'source': 'cache'})}\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # ── Layer 2b: ChromaDB + TF-IDF retrieval ────────────
    clean_query = normalize_query(question)
    log.info("ChromaDB retrieval for: %s", clean_query[:60])

    loop = asyncio.get_event_loop()

    semantic, keyword = await asyncio.gather(
        loop.run_in_executor(None, lambda: semantic_search(clean_query)),
        loop.run_in_executor(None, lambda: keyword_search(clean_query)),
    )

    fused   = fuse(semantic, keyword)
    context = format_context(fused)

    if not context:
        log.warning("No context found — LLM will answer from rules only")

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:]
    )

    prompt_text = PROMPT.format(
        context=context or "No relevant context found.",
        history=history_text,
        question=question,
    )

    # ── Layer 3: Stream LLM response ─────────────────────
    async def stream():
        full = ""

        async for token in llm.astream(prompt_text):
            full += token
            yield f"data: {json.dumps({'content': token, 'source': 'chroma'})}\n\n"

        # Update session memory
        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant",  "content": full})
        if len(history) > HISTORY_WINDOW * 2:
            session_store[session_id] = history[-(HISTORY_WINDOW * 2):]

        # Store in response cache
        if len(_response_cache) >= CACHE_MAX:
            _response_cache.pop(next(iter(_response_cache)))
        _response_cache[ck] = full

    return StreamingResponse(stream(), media_type="text/event-stream")

# ─────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "chunks":       len(hybrid_data["texts"]) if hybrid_data else 0,
        "json_intents": len(_INTENT_INDEX),
        "cache_size":   len(_response_cache),
        "embed_cache":  len(_embed_cache),
    }

# ─────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=False)