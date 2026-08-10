
import os,json,asyncio,logging,pickle,hashlib,re,random,uuid
from difflib import SequenceMatcher
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_core.prompts import PromptTemplate

os.environ["ANONYMIZED_TELEMETRY"]="False"
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
log=logging.getLogger("kalaw")
app=FastAPI()

app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR,"data")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")
HYBRID_PATH=os.path.join(DATA_DIR,"tfidf_embeddings.pkl")
JSON_PATH=os.path.join(DATA_DIR,"faculty_manual.json")

JSON_SCORE_THRESHOLD=0.65
CHROMA_TOP_K=6
CHROMA_FETCH_K=16
KEYWORD_TOP_K=6
MAX_CONTEXT_CHUNKS=5
MAX_CHUNK_CHARS=1200
RERANK_MIN_SCORE=0.08
CACHE_MAX=150
EMBED_CACHE_MAX=500
HISTORY_WINDOW=4
TYPO_MIN_WORD_LENGTH=4
TYPO_THRESHOLD=0.84
FOLLOWUP_HISTORY_MESSAGES=4

# ============================================================
# DATASET
# ============================================================

INTENTS=[]
if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH,"r",encoding="utf-8") as f:
            dataset=json.load(f)
        if isinstance(dataset,dict):
            INTENTS=dataset.get("intents",[])
        elif isinstance(dataset,list):
            INTENTS=dataset
        else:
            log.error("Unsupported JSON dataset format.")
    except Exception as e:
        log.exception("Failed to load JSON dataset: %s",e)
else:
    log.warning("Dataset not found: %s",JSON_PATH)

_INTENT_INDEX=[]
_VOCABULARY=set()

for intent in INTENTS:
    intent_name=intent.get("intent","")
    category=intent.get("category","")
    patterns=intent.get("patterns",[]) or []
    keywords=intent.get("keywords",[]) or []
    aliases=intent.get("aliases",[]) or []
    response=intent.get("response","")
    source=intent.get("source",{}) or {}

    for phrase in patterns+aliases:
        if not isinstance(phrase,str):
            continue
        phrase_lower=phrase.lower().strip()
        tokens=set(re.findall(r"[a-z0-9]+",phrase_lower))
        _INTENT_INDEX.append({
            "intent":intent_name,
            "category":category,
            "pattern":phrase_lower,
            "tokens":tokens,
            "keywords":keywords,
            "aliases":aliases,
            "response":response,
            "source":source
        })
        _VOCABULARY.update(tokens)

    for keyword in keywords:
        if isinstance(keyword,str):
            _VOCABULARY.update(re.findall(r"[a-z0-9]+",keyword.lower()))

log.info("JSON intents loaded: %d",len(INTENTS))
log.info("JSON patterns/aliases indexed: %d",len(_INTENT_INDEX))
log.info("Typo vocabulary loaded: %d",len(_VOCABULARY))

# ============================================================
# GREETINGS
# ============================================================

_GREETING_TRIGGERS={
    "hi","hello","hey","goodmorning","goodafternoon","goodevening",
    "good day","howdy","greetings","hi there","hello there","hey there",
    "what's up","sup","yo","kamusta","kumusta","musta",
    "magandang umaga","magandang hapon","magandang gabi","magandang araw",
    "maayong buntag","maayong udto","maayong hapon","maayong gabii","kumusta ka"
}

_GREETING_RESPONSES=[
    "Hello! I'm **KALAW**, your CPSU Faculty Manual assistant.\n\nI can help you with policies, procedures, leave benefits, faculty ranks, and anything covered in the Faculty Manual. What would you like to know?",
    "Hi there! Welcome — I'm **KALAW**, the CPSU Faculty Manual chatbot.\n\nFeel free to ask me about faculty policies, duties, benefits, or any section of the manual. How can I assist you today?",
    "Good day! I'm **KALAW**, here to help you navigate the CPSU Faculty Manual.\n\nAsk me anything about faculty rules, leave entitlements, promotions, or academic procedures. What's your question?",
    "Hey! I'm **KALAW** — your go-to guide for the CPSU Faculty Manual.\n\nWhether it's about teaching loads, leave policies, or faculty obligations, I'm ready to help. What do you need?"
]

def greeting_match(query:str)->str|None:
    q=re.sub(r"[^a-z0-9\s']"," ",query.lower()).strip()
    q=re.sub(r"\s+"," ",q)
    if q in _GREETING_TRIGGERS:
        return random.choice(_GREETING_RESPONSES)
    for trigger in _GREETING_TRIGGERS:
        if q.startswith(trigger+" "):
            return random.choice(_GREETING_RESPONSES)
    return None

# ============================================================
# NORMALIZATION / TYPO CORRECTION
# ============================================================

def normalize_query(q:str)->str:
    q=str(q).lower().replace("’","'")
    q=re.sub(r"[^a-z0-9\s']"," ",q)
    return re.sub(r"\s+"," ",q).strip()

def correct_common_typos(query:str)->str:
    normalized=normalize_query(query)
    if not normalized:
        return normalized
    corrected=[]
    for word in normalized.split():
        if word in _VOCABULARY or len(word)<TYPO_MIN_WORD_LENGTH:
            corrected.append(word)
            continue
        best_word=word
        best_score=0.0
        for candidate in _VOCABULARY:
            if abs(len(candidate)-len(word))>2:
                continue
            if candidate[0]!=word[0] and candidate[-1]!=word[-1]:
                continue
            score=SequenceMatcher(None,word,candidate).ratio()
            if score>best_score:
                best_score=score
                best_word=candidate
        corrected.append(best_word if best_score>=TYPO_THRESHOLD else word)
    return " ".join(corrected)

# ============================================================
# FOLLOW-UP QUESTION HANDLING
# ============================================================

_FOLLOWUP_TRIGGERS={
    "what about","how about","what if","and what about","and how about",
    "what are the requirements","what are the rules","what is the requirement",
    "what is the rule","how does that work","how does this work",
    "can you explain","explain more","tell me more","more details",
    "what about that","how about that","what about this","how about this",
    "and then","what else","is that allowed","is this allowed",
    "how many","how much","when can","who can","who is eligible"
}

def is_followup_question(query:str)->bool:
    q=normalize_query(query)
    words=q.split()

    if any(q.startswith(trigger) for trigger in _FOLLOWUP_TRIGGERS):
        return True

    if len(words)<=7:
        followup_words={
            "that","this","those","these","it","they","them",
            "requirements","requirement","rules","rule","limit",
            "limits","process","procedure","benefits","eligibility",
            "eligible","allowed","allow","duration","amount","steps"
        }
        if any(w in followup_words for w in words):
            return True

    return False

def build_retrieval_query(question:str,history:list)->str:
    clean_question=correct_common_typos(question)

    if not history or not is_followup_question(question):
        return clean_question

    previous=[]
    for message in history[-FOLLOWUP_HISTORY_MESSAGES:]:
        content=message.get("content","").strip()
        if not content:
            continue
        previous.append(f"{message.get('role','user')}: {content}")

    if not previous:
        return clean_question

    history_text=" ".join(previous)
    combined=f"{history_text} {clean_question}"
    combined=normalize_query(combined)

    # Keep retrieval query reasonably small.
    words=combined.split()
    if len(words)>80:
        combined=" ".join(words[-80:])

    log.info("Follow-up retrieval query: %s",combined[:120])
    return combined

# ============================================================
# JSON INTENT MATCHING
# ============================================================

def json_match(query:str)->dict|None:
    q_norm=correct_common_typos(query)
    q_tokens=set(re.findall(r"[a-z0-9]+",q_norm))
    if not q_tokens:return None

    best_score=0.0
    best_match=None

    for entry in _INTENT_INDEX:
        pattern=entry["pattern"]
        tokens=entry["tokens"]
        if not tokens:continue

        overlap=len(q_tokens&tokens)/max(1,len(q_tokens|tokens))
        sequence=SequenceMatcher(None,q_norm,pattern).ratio()

        keyword_hits=0
        for keyword in entry["keywords"]:
            k=normalize_query(keyword)
            if k and k in q_norm:keyword_hits+=1

        alias_hits=0
        for alias in entry["aliases"]:
            a=normalize_query(alias)
            if a and a in q_norm:alias_hits+=1

        score=overlap*0.35+sequence*0.30
        score+=min(keyword_hits*0.10,0.30)
        score+=min(alias_hits*0.10,0.20)

        if pattern in q_norm:score+=0.20

        # Strong boost for important multi-word concepts.
        q_text=f" {q_norm} "
        for phrase in ("part time","teaching load","part-time",
                       "faculty workload","overload","leave",
                       "promotion","tenure","salary"):
            if phrase in q_text and phrase in normalize_query(
                " ".join(entry["keywords"]+entry["aliases"])
            ):
                score+=0.10

        score=min(score,1.0)

        if score>best_score:
            best_score=score
            best_match=entry

    if best_match and best_score>=JSON_SCORE_THRESHOLD:
        return {
            "intent":best_match["intent"],
            "category":best_match["category"],
            "response":best_match["response"],
            "source":best_match["source"],
            "score":round(best_score,4),
            "corrected_query":q_norm
        }

    return None

# ============================================================
# OLLAMA
# ============================================================

embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
llm=OllamaLLM(
    model="llama3.2:latest",
    temperature=0,
    num_ctx=2048,
    num_predict=350,
    repeat_penalty=1.1,
    top_k=20,
    top_p=0.85
)

# ============================================================
# CHROMA / TF-IDF
# ============================================================

vector_db=Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

hybrid_data=None
if os.path.exists(HYBRID_PATH):
    try:
        with open(HYBRID_PATH,"rb") as f:
            hybrid_data=pickle.load(f)
        log.info("TF-IDF index loaded: %d documents",len(hybrid_data["texts"]))
    except Exception as e:
        log.exception("Failed to load TF-IDF index: %s",e)
else:
    log.warning("TF-IDF index not found: %s",HYBRID_PATH)

# ============================================================
# SESSION / CACHE
# ============================================================

session_store={}
_response_cache={}
_embed_cache={}

def new_session_id():
    return str(uuid.uuid4())

def cache_key(q:str)->str:
    return hashlib.md5(normalize_query(q).encode("utf-8")).hexdigest()

def get_or_embed(query:str)->list[float]:
    ck=cache_key(query)
    if ck not in _embed_cache:
        _embed_cache[ck]=embeddings.embed_query(query)
        if len(_embed_cache)>EMBED_CACHE_MAX:
            _embed_cache.pop(next(iter(_embed_cache)))
    return _embed_cache[ck]

# ============================================================
# PROMPT
# ============================================================

PROMPT=PromptTemplate(
    input_variables=["context","history","question"],
    template="""You are KALAW, a strict CPSU Faculty Manual assistant.

RULES:
- Use ONLY the provided CPSU Faculty Manual context.
- Do NOT guess, invent, or hallucinate policies, numbers, requirements, dates, or benefits.
- Use conversation history only to understand what the user is referring to.
- If the current question cannot be answered from the provided context, say exactly:
"Not found in the Faculty Manual."
- Answer directly, naturally, and conversationally.
- If the user asks two or more things, answer every part separately and do not omit a part.
- Use clear headings or bullets for multi-part questions.
- Preserve exact numbers, requirements, conditions, dates, rates, units, and limits from the context.
- If source/article/section/page information is available, mention it.
- Do not claim information is in the Faculty Manual if it is not in the context.
- For follow-up questions, use the previous conversation only to resolve references such as "this", "that", "it", "they", "them", "their", or "what about".
- Rewrite short follow-up questions mentally into a complete retrieval question before answering.
- Do not copy unrelated information from the conversation history.
- If only part of a multi-part question is supported by the context, answer the supported part and say "Not found in the Faculty Manual." for the unsupported part.

CONTEXT:
{context}

HISTORY:
{history}

QUESTION:
{question}

ANSWER:"""
)

# ============================================================
# RETRIEVAL
# ============================================================

def keyword_search(query:str,top_k:int=KEYWORD_TOP_K)->list[dict]:
    if not hybrid_data:
        return []
    try:
        vec=hybrid_data["vectorizer"].transform([query])
        scores=(hybrid_data["tfidf"]*vec.T).toarray().flatten()
        top_idx=scores.argsort()[-top_k:][::-1]
        return [
            {
                "text":hybrid_data["texts"][i],
                "metadata":hybrid_data["metadatas"][i],
                "score":float(scores[i])
            }
            for i in top_idx if scores[i]>0.01
        ]
    except Exception as e:
        log.warning("Keyword search failed: %s",e)
        return []

def semantic_search(query:str)->list:
    try:
        return vector_db.max_marginal_relevance_search_by_vector(
            get_or_embed(query),
            k=CHROMA_TOP_K,
            fetch_k=CHROMA_FETCH_K,
            lambda_mult=0.6
        )
    except Exception as e:
        log.warning("MMR search failed: %s",e)
        try:
            return vector_db.similarity_search(query,k=CHROMA_TOP_K)
        except Exception as inner:
            log.error("Chroma retrieval failed: %s",inner)
            return []

def fuse(semantic_docs:list,keyword_docs:list[dict],k:int=60)->list[dict]:
    scores={}
    store={}

    for i,doc in enumerate(semantic_docs):
        text=doc.page_content
        key=text[:200]
        scores[key]=scores.get(key,0)+1/(k+i+1)
        store[key]={"text":text,"metadata":doc.metadata}

    for i,doc in enumerate(keyword_docs):
        text=doc["text"]
        key=text[:200]
        scores[key]=scores.get(key,0)+1/(k+i+1)
        if key not in store:
            store[key]={"text":text,"metadata":doc.get("metadata",{})}

    return sorted(
        store.values(),
        key=lambda x:scores.get(x["text"][:200],0),
        reverse=True
    )

def rerank(query:str,docs:list[dict],top_k:int=MAX_CONTEXT_CHUNKS)->list[dict]:
    q=normalize_query(query)
    q_tokens=set(re.findall(r"[a-z0-9]+",q))
    if not docs:return []
    ranked=[]
    for rank,doc in enumerate(docs):
        text=normalize_query(doc.get("text",""))
        if not text:continue
        d_tokens=set(re.findall(r"[a-z0-9]+",text))
        overlap=len(q_tokens&d_tokens)/max(1,len(q_tokens))
        phrase_hits=sum(1 for t in q_tokens if len(t)>3 and t in text)
        phrase_score=min(phrase_hits/max(1,len(q_tokens)),1.0)
        rank_score=1/(rank+1)
        score=overlap*0.55+phrase_score*0.25+rank_score*0.20
        item=dict(doc); item["_rerank_score"]=score
        if score>=RERANK_MIN_SCORE:ranked.append(item)
    ranked.sort(key=lambda x:x["_rerank_score"],reverse=True)
    log.info("Reranked %d candidates -> %d chunks",len(docs),min(len(ranked),top_k))
    return ranked[:top_k]

_BAD_KEYWORDS={"table of contents","index","copyright","acknowledgement"}

def format_context(docs:list[dict],max_chunks:int=MAX_CONTEXT_CHUNKS)->str:
    seen=set()
    out=[]

    for doc in docs:
        text=doc["text"].strip()

        if len(text)<100:
            continue

        if any(bad in text.lower() for bad in _BAD_KEYWORDS):
            continue

        key=text[:120]
        if key in seen:
            continue

        seen.add(key)
        metadata=doc.get("metadata") or {}
        parts=[]

        if metadata.get("page"):
            parts.append(f"Page {metadata['page']}")
        if metadata.get("article"):
            parts.append(f"Article {metadata['article']}")
        if metadata.get("section"):
            parts.append(f"Section {metadata['section']}")

        prefix=f"[{' | '.join(parts)}]\n" if parts else ""
        out.append(prefix+text[:MAX_CHUNK_CHARS])

        if len(out)>=max_chunks:
            break

    return "\n\n---\n\n".join(out)

def sse(data:dict)->str:
    return f"data: {json.dumps(data,ensure_ascii=False)}\n\n"

# ============================================================
# NEW SESSION
# ============================================================

@app.get("/api/session/new")
def create_session():
    session_id=new_session_id()
    session_store[session_id]=[]
    return {"session_id":session_id}

# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
async def chat(request:Request):
    data=await request.json()
    question=str(data.get("question","")).strip()
    session_id=str(data.get("session_id") or data.get("chat_id") or "").strip()

    if not session_id:
        session_id=new_session_id()

    if not question:
        return {"error":"Empty question","session_id":session_id}

    if session_id not in session_store:
        session_store[session_id]=[]

    history=session_store[session_id]

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    greeting=greeting_match(question)

    if greeting:
        history.append({"role":"user","content":question})
        history.append({"role":"assistant","content":greeting})

        if len(history)>HISTORY_WINDOW*2:
            session_store[session_id]=history[-(HISTORY_WINDOW*2):]

        async def greeting_stream():
            yield sse({
                "content":greeting,
                "source":"greeting",
                "session_id":session_id
            })

        return StreamingResponse(
            greeting_stream(),
            media_type="text/event-stream"
        )

    # --------------------------------------------------------
    # TYPO CORRECTION
    # --------------------------------------------------------

    clean_question=correct_common_typos(question)

    if clean_question!=normalize_query(question):
        log.info(
            "Typo correction: '%s' -> '%s'",
            question[:80],
            clean_question[:80]
        )

    # --------------------------------------------------------
    # FOLLOW-UP DETECTION
    # --------------------------------------------------------

    followup=is_followup_question(question)

    if followup:
        log.info("Follow-up question detected | session=%s",session_id)

    # --------------------------------------------------------
    # DIRECT JSON INTENT
    # --------------------------------------------------------
    # Do not let a generic JSON intent override a contextual follow-up.

    json_answer=None if followup else json_match(question)

    if json_answer:
        log.info(
            "JSON intent hit: %s | score=%.3f",
            json_answer["intent"],
            json_answer["score"]
        )

        history.append({
            "role":"user",
            "content":question
        })
        history.append({
            "role":"assistant",
            "content":json_answer["response"]
        })

        if len(history)>HISTORY_WINDOW*2:
            session_store[session_id]=history[-(HISTORY_WINDOW*2):]

        async def fast_stream():
            yield sse({
                "content":json_answer["response"],
                "source":"json",
                "intent":json_answer["intent"],
                "category":json_answer["category"],
                "metadata":json_answer["source"],
                "score":json_answer["score"],
                "corrected_query":json_answer["corrected_query"],
                "session_id":session_id
            })

        return StreamingResponse(
            fast_stream(),
            media_type="text/event-stream"
        )

    # --------------------------------------------------------
    # FOLLOW-UP-AWARE RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query=build_retrieval_query(
        clean_question,
        history
    )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------
    # Follow-ups depend on conversation context, so include recent
    # history in the cache key to prevent cross-conversation matches.

    if followup:
        history_key="|".join(
            f"{m['role']}:{m['content']}"
            for m in history[-FOLLOWUP_HISTORY_MESSAGES:]
        )
        ck=cache_key(history_key+"|"+retrieval_query)
    else:
        ck=cache_key(retrieval_query)

    if ck in _response_cache:
        cached=_response_cache[ck]

        async def cached_stream():
            yield sse({
                "content":cached,
                "source":"cache",
                "session_id":session_id
            })

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream"
        )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_text="\n".join(
        f"{m['role']}: {m['content']}"
        for m in history[-HISTORY_WINDOW:]
    )

    # --------------------------------------------------------
    # HYBRID RETRIEVAL
    # --------------------------------------------------------

    log.info(
        "Hybrid retrieval | session=%s | query=%s",
        session_id,
        retrieval_query[:120]
    )

    loop=asyncio.get_running_loop()

    semantic_task=loop.run_in_executor(
        None,
        lambda:semantic_search(retrieval_query)
    )

    keyword_task=loop.run_in_executor(
        None,
        lambda:keyword_search(retrieval_query)
    )

    semantic,keyword=await asyncio.gather(
        semantic_task,
        keyword_task
    )

    fused=fuse(semantic,keyword)
    reranked=rerank(retrieval_query,fused,MAX_CONTEXT_CHUNKS)
    context=format_context(reranked,MAX_CONTEXT_CHUNKS)

    if not context:
        log.warning(
            "No context found for: %s",
            retrieval_query[:100]
        )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    prompt_text=PROMPT.format(
        context=context or "No relevant context found.",
        history=history_text or "No previous conversation.",
        question=question
    )

    async def stream():
        full=""

        try:
            async for token in llm.astream(prompt_text):
                full+=token

                yield sse({
                    "content":token,
                    "source":"chroma",
                    "session_id":session_id,
                    "follow_up":followup
                })

        except Exception as e:
            log.exception(
                "LLM generation failed: %s",
                e
            )

            yield sse({
                "content":"Sorry, I couldn't generate an answer right now. Please make sure Ollama is running.",
                "source":"error",
                "session_id":session_id
            })

            return

        # ----------------------------------------------------
        # SAVE CONVERSATION
        # ----------------------------------------------------

        history.append({
            "role":"user",
            "content":question
        })

        history.append({
            "role":"assistant",
            "content":full
        })

        if len(history)>HISTORY_WINDOW*2:
            session_store[session_id]=history[-(HISTORY_WINDOW*2):]

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        if len(_response_cache)>=CACHE_MAX:
            _response_cache.pop(next(iter(_response_cache)))

        _response_cache[ck]=full

    return StreamingResponse(
        stream(),
        media_type="text/event-stream"
    )

# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/api/session/{session_id}")
def delete_session(session_id:str):
    if session_id in session_store:
        del session_store[session_id]

    return {
        "status":"deleted",
        "session_id":session_id
    }

# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {
        "status":"ok",
        "active_sessions":len(session_store),
        "dataset_intents":len(INTENTS),
        "json_patterns":len(_INTENT_INDEX),
        "vocabulary_words":len(_VOCABULARY),
        "chunks":len(hybrid_data["texts"]) if hybrid_data else 0,
        "cache_size":len(_response_cache),
        "embed_cache":len(_embed_cache),
        "model":"llama3.2:latest",
        "embedding_model":"mxbai-embed-large:latest","reranking":True
    }

# ============================================================
# RUN
# ============================================================

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="127.0.0.1",port=5000,reload=False)

