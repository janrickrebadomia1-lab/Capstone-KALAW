import os,json,asyncio,logging,pickle,hashlib,re,uuid
from difflib import SequenceMatcher
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import httpx
from langchain_core.prompts import PromptTemplate
from greetings import greeting_match
import random

os.environ["ANONYMIZED_TELEMETRY"]="False"
logging.basicConfig(level=logging.INFO,format="%(levelname)s: %(message)s")
log=logging.getLogger("kalaw")
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://capstone-kalaw.vercel.app",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(BASE_DIR,"data")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")
HYBRID_PATH=os.path.join(DATA_DIR,"tfidf_embeddings.pkl")
JSON_PATH=os.path.join(DATA_DIR,"faculty_manual.json")

JSON_SCORE_THRESHOLD=0.90
CHROMA_TOP_K=4
CHROMA_FETCH_K=8
KEYWORD_TOP_K=6
MAX_CONTEXT_CHUNKS=4
MAX_CHUNK_CHARS=1700
CACHE_MAX=200
EMBED_CACHE_MAX=750
RETRIEVAL_CACHE_MAX=300
KEYWORD_FAST_MIN_OVERLAP=0.60
KEYWORD_FAST_MIN_BM25=0.20
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
_INTENT_FAST_INDEX=[]

for intent in INTENTS:
    intent_name=intent.get("intent","")
    category=intent.get("category","")
    patterns=intent.get("patterns",[]) or []
    keywords=intent.get("keywords",[]) or []
    aliases=intent.get("aliases",[]) or []
    response=intent.get("response","")
    source=intent.get("source",{}) or {}

    # Compact intent-level index used for the fast JSON gate.
    _INTENT_FAST_INDEX.append({
        "intent": intent_name,
        "category": category,
        "keywords": keywords,
        "aliases": aliases, 
        "response": response,
        "source": source,
        "patterns": [p for p in patterns if isinstance(p,str)],
    })

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

    # Use user messages as the conversational subject. Assistant answers are
    # deliberately excluded so their wording cannot contaminate retrieval.
    user_messages=[
        m.get("content","").strip()
        for m in history
        if m.get("role")=="user" and m.get("content","").strip()
    ]

    if not user_messages:
        return clean_question

    previous_user=user_messages[-3:]

    # Prefer the most recent explicit faculty-manual topic/subject.
    subject=""
    topic_words={
        "benefits","benefit","compensation","salary","salaries","pay","payment",
        "teaching load","workload","leave","promotion","tenure","qualification",
        "qualifications","requirements","duties","responsibilities","mission",
        "vision","allowance","stipend","privilege","privileges"
    }

    for text in reversed(previous_user):
        n=normalize_query(text)
        if any(term in n for term in topic_words):
            subject=text
            break

    if not subject:
        subject=previous_user[-1]

    # If the current follow-up contains a topic, combine it with the previous
    # subject. This makes "What about their benefits?" become a focused query.
    current=clean_question
    current_n=normalize_query(current)

    # Remove conversational pronouns from the current question only.
    current_n=re.sub(
        r"\b(their|them|they|it|this|that|those|these)\b",
        " ",
        current_n
    )
    current_n=re.sub(r"\s+"," ",current_n).strip()

    combined=f"{subject} {current_n}".strip()
    combined=normalize_query(combined)

    words=combined.split()
    if len(words)>50:
        combined=" ".join(words[-50:])

    log.info("Follow-up retrieval query: %s",combined[:160])
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

OLLAMA_BASE_URL=os.getenv("OLLAMA_BASE_URL","http://127.0.0.1:11434")
OLLAMA_MODEL="qwen3:8b"
OLLAMA_TIMEOUT=None
OLLAMA_OPTIONS = {
    "temperature": 0,
    "num_ctx": 3072,
    "num_predict": 220,
    "repeat_penalty": 1.1,
    "top_k": 20,
    "top_p": 0.85
}

OLLAMA_KEEP_ALIVE = "15m"

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
_retrieval_cache={}

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

def log_elapsed(label:str, started:float)->None:
    elapsed=asyncio.get_running_loop().time()-started
    log.info("%s: %.3fs", label, elapsed)

# ============================================================
# PROMPT
# ============================================================

PROMPT=PromptTemplate(
    input_variables=["context","history","question"],
    template="""You are KALAW, a strict CPSU Faculty Manual assistant.

RULES:
- Understand the user's question regardless of whether it is written in English, Cebuano/Bisaya, Filipino, or a natural mixture of these languages.
- ALWAYS provide the final answer in ENGLISH. Never answer factual Faculty Manual questions in Cebuano/Bisaya, Filipino, Waray, or another language.
- The user's language/dialect affects understanding and retrieval only; it does not determine the final response language.
- Use ONLY the provided CPSU Faculty Manual context.
- Do NOT guess, invent, or hallucinate policies, numbers, requirements, dates, or benefits.
- Use conversation history only to understand what the user is referring to.
- If the current question cannot be answered from the provided context, say exactly:
"Not found in the Faculty Manual."
- If the context contains a direct statement that answers the question, use that
statement even if another retrieved passage is less specific.
- Never say "Not found in the Faculty Manual" when the context contains direct
evidence for the requested topic.
- Do not turn a specific benefit (such as leave benefits) into a claim that it is
the complete list of all benefits.
- Answer directly, naturally, and conversationally.
- Treat every requested topic as a separate information request.
- Answer EVERY part of a multi-part question; never answer only one topic.
- Retrieve and answer each distinct topic from its matching evidence.
- Do not use evidence for one topic (for example, compensation) as evidence for
a different topic (for example, benefits).
- Use clear headings or bullets when there are multiple requested topics.
- Base each factual statement on the matching evidence in CONTEXT.
- Do not use a generic faculty-definition passage to answer a specific question about duties, benefits, workload, compensation, qualifications, or requirements unless it directly answers that question.
- Preserve exact numbers, requirements, conditions, dates, rates, units, and limits from the context.
- When the context contains a complete list of items for the requested topic,
include all supported items rather than silently dropping an entry.
- If source/article/section/page information is available, mention it.
- Do not claim information is in the Faculty Manual if it is not in the context.
- For follow-up questions, use conversation history only to resolve references such as "this", "that", "it", "they", "them", "their", or "what about".
- If only one part of a multi-part question is supported, answer that part and say "Not found in the Faculty Manual." for the unsupported part.
- Do not copy unrelated information from conversation history.

CONTEXT:
{context}

HISTORY:
{history}

QUESTION:
{question}

ANSWER:"""
)

# ============================================================
# FAST JSON INTENT GATE
# ============================================================

def fast_json_match(query:str)->dict|None:
    """High-confidence JSON match for direct, curated answers.

    This is intentionally conservative. A direct JSON response is only returned
    when the intent evidence is substantially stronger than the alternatives.
    Lower-confidence queries continue through BM25/Chroma.
    """
    q=correct_common_typos(query)
    q_norm=normalize_query(q)
    q_tokens=set(re.findall(r"[a-z0-9]+",q_norm))

    if not q_tokens:
        return None

    best=None
    second=0.0

    for intent in _INTENT_FAST_INDEX:
        keywords=intent["keywords"] or []
        aliases=intent["aliases"] or []
        patterns=intent["patterns"] or []

        # Use representative patterns/aliases only for the fast gate.
        candidates=patterns[:12] + aliases[:8]
        if not candidates:
            continue

        best_entry_score=0.0
        best_phrase=""

        for phrase in candidates:
            p=normalize_query(phrase)
            if not p:
                continue

            p_tokens=set(re.findall(r"[a-z0-9]+",p))
            if not p_tokens:
                continue

            overlap=len(q_tokens & p_tokens) / max(1,len(q_tokens | p_tokens))
            sequence=SequenceMatcher(None,q_norm,p).ratio()

            keyword_hits=sum(
                1 for k in keywords
                if isinstance(k,str)
                and (normalize_query(k) in q_norm)
            )

            alias_hits=sum(
                1 for a in aliases
                if isinstance(a,str)
                and (normalize_query(a) in q_norm)
            )

            score=(
                overlap*0.42
                + sequence*0.28
                + min(keyword_hits*0.08,0.20)
                + min(alias_hits*0.08,0.16)
            )

            if p==q_norm:
                score=1.0

            if score>best_entry_score:
                best_entry_score=score
                best_phrase=p

        if best_entry_score>second:
            second=best_entry_score

        if best is None or best_entry_score>best["score"]:
            best={
                "intent":intent["intent"],
                "category":intent["category"],
                "response":intent["response"],
                "source":intent["source"],
                "score":min(best_entry_score,1.0),
                "matched_phrase":best_phrase,
                "corrected_query":q_norm,
            }

    if not best:
        return None

    margin=best["score"]-second

    # Conservative threshold to protect retrieval accuracy.
    if best["score"]>=0.94 and margin>=0.05:
        return best

    return None


# ============================================================
# RETRIEVAL
# ============================================================

def keyword_search(query:str,top_k:int=KEYWORD_TOP_K)->list[dict]:
    if not hybrid_data or "bm25" not in hybrid_data:
        return []
    try:
        bm25=hybrid_data["bm25"]
        terms=set(re.findall(r"[a-z0-9]+",normalize_query(query)))
        if not terms:return []
        scores={}
        lengths=bm25["doc_lengths"]; avg=float(bm25["avg_doc_len"] or 1.0); k1=float(bm25.get("k1",1.5)); b=float(bm25.get("b",0.75))
        for term in terms:
            posting=bm25["postings"].get(term)
            if not posting: continue
            term_idf=float(bm25["idf"].get(term,0.0))
            for doc_id,tf in posting.items():
                norm=(1-b)+b*(lengths[doc_id]/(avg+1e-9))
                scores[doc_id]=scores.get(doc_id,0.0)+term_idf*((tf*(k1+1.0))/(tf+k1*norm+1e-9))
        top_ids=sorted(scores,key=scores.get,reverse=True)[:top_k]
        return [{"text":hybrid_data["texts"][i],"metadata":hybrid_data["metadatas"][i] or {},"score":float(scores[i])} for i in top_ids if scores[i]>0]
    except Exception as e:
        log.warning("BM25 search failed: %s",e)
        return []

def keyword_confidence(query:str,docs:list[dict])->float:
    if not docs:return 0.0
    q_words=_expand_topics(_topic_words(query))
    if not q_words:return 0.0
    top=float(docs[0].get("score",0.0)); second=float(docs[1].get("score",0.0)) if len(docs)>1 else 0.0
    margin=min(max((top-second)/(top+1e-9),0.0),1.0)
    best=0.0
    for doc in docs[:3]:
        text=normalize_query(doc.get("text","")); meta=doc.get("metadata") or {}
        meta_text=normalize_query(" ".join(str(meta.get(k,"")) for k in ("heading","section","article","chapter")))
        body=_topic_words(text); head=_topic_words(meta_text)
        overlap=max(len(q_words&body)/max(1,len(q_words)),len(q_words&head)/max(1,len(q_words)))
        phrase=0.0; qn=normalize_query(query)
        for p in ("faculty workload","teaching load","part time faculty","research load","leave of absence","promotion","tenure","faculty benefits","salary"):
            if p in qn and p in text: phrase=0.20; break
        score=min(1.0,overlap*0.60+min(top/8.0,1.0)*0.25+margin*0.15+phrase)
        best=max(best,score)
    return best

def semantic_search(query:str)->list:
    """Embedding retrieval is used only when lexical retrieval is insufficient."""
    try:
        return vector_db.similarity_search_by_vector(
            get_or_embed(query),
            k=CHROMA_TOP_K
        )
    except Exception as e:
        log.error("Chroma retrieval failed: %s",e)
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

_GENERIC_TERMS=set("""what is are the a an of for to in on and or how why who where when
can could would should do does did tell me about please me faculty faculties
member members university cpsu""".split())

def _topic_words(text:str)->set[str]:
    words=set(re.findall(r"[a-z0-9]+",normalize_query(text)))
    return {w for w in words if len(w)>2 and w not in _GENERIC_TERMS}

def _expand_topics(words:set[str])->set[str]:
    expanded=set(words)
    groups=[
        {"benefit","benefits","incentive","incentives","allowance","allowances"},
        {"duty","duties","responsibility","responsibilities"},
        {"qualification","qualifications","requirement","requirements"},
        {"mission"},
        {"vision"},
        {"compensation","salary","salaries","pay","rate","rates"},
        {"teaching","load","workload","overload"},
        {"leave","leaves"},
        {"promotion","promotions","tenure"},
    ]
    for group in groups:
        if expanded&group: expanded.update(group)
    return expanded

def split_question_parts(query:str)->list[str]:
    q=correct_common_typos(query)
    # Separate explicit multiple questions first.
    parts=[x.strip() for x in re.split(r"\?\s*|\n+",q) if x.strip()]
    if len(parts)>1:return parts

    # Handle "What are X and Y of Z?" / "What is X and Y of Z?"
    m=re.match(r"^(what\s+(?:is|are)|tell\s+me\s+about)\s+(.+?)\s+(?:of|for)\s+(.+)$",q)
    if m:
        prefix,topics,subject=m.groups()
        pieces=[x.strip() for x in re.split(r"\s*,\s*|\s+and\s+|\s+or\s+",topics) if x.strip()]
        if len(pieces)>1 and all(len(x.split())<=7 for x in pieces):
            return [f"{pieces[0]} of {subject}"]+[f"{x} of {subject}" for x in pieces[1:]]

    # Handle "What is the mission and vision of CPSU?" specifically and
    # similar two-topic questions where "of/for" occurs after both topics.
    m=re.match(r"^(what\s+(?:is|are)|tell\s+me\s+about)\s+(.+?)\s+(and|or)\s+(.+?)\s+(of|for)\s+(.+)$",q)
    if m:
        prefix,a,conj,b,prep,subject=m.groups()
        if len(a.split())<=6 and len(b.split())<=6:
            return [f"{a} {prep} {subject}",f"{b} {prep} {subject}"]

    return [q]

def rerank_candidates(query:str,docs:list[dict],limit:int=MAX_CONTEXT_CHUNKS)->list[dict]:
    q_words=_expand_topics(_topic_words(query))
    if not docs:return []
    ranked=[]
    for d in docs:
        text=normalize_query(d.get("text",""))
        meta=d.get("metadata") or {}
        meta_text=normalize_query(" ".join(
            str(meta.get(k,"")) for k in
            ("heading","title","section","article","category","source")
        ))
        body_words=_topic_words(text)
        head_words=_topic_words(meta_text)
        body_overlap=len(q_words&body_words)/max(1,len(q_words))
        head_overlap=len(q_words&head_words)/max(1,len(q_words))
        exact_hits=sum(1 for w in q_words if w in text)
        exact_score=min(exact_hits/max(1,len(q_words)),1.0)
        # RRF rank is retained by fuse; use it as a weak prior only.
        base=float(d.get("_rrf",0.0))
        score=body_overlap*0.42+head_overlap*0.30+exact_score*0.23+min(base*10,0.05)
        # A chunk that contains none of the actual topic words is almost
        # certainly unrelated, even if it says "faculty" many times.
        if q_words and not (q_words&body_words) and not (q_words&head_words):
            score=0.0
        item=dict(d); item["_relevance"]=score
        ranked.append(item)
    ranked.sort(key=lambda x:x["_relevance"],reverse=True)
    return [x for x in ranked if x["_relevance"]>0][:limit]

def fast_intent_match(query:str)->dict|None:
    q=correct_common_typos(query); q_tokens=set(re.findall(r"[a-z0-9]+",q))
    if not q_tokens:return None
    best=None; best_score=0.0
    for intent in INTENTS:
        candidates=(intent.get("patterns",[]) or [])[:10]+(intent.get("aliases",[]) or [])[:6]
        kws=intent.get("keywords",[]) or []
        for phrase in candidates:
            p=normalize_query(phrase); p_tokens=set(re.findall(r"[a-z0-9]+",p))
            if not p_tokens: continue
            overlap=len(q_tokens&p_tokens)/max(1,len(q_tokens|p_tokens))
            score=overlap+min(sum(1 for k in kws if normalize_query(k) and normalize_query(k) in q)*0.05,0.15)
            if p==q: score=1.0
            if score>best_score: best_score=score; best={"intent":intent.get("intent",""),"category":intent.get("category",""),"response":intent.get("response",""),"source":intent.get("source",{}),"score":min(score,1.0)}
    return best if best and best["score"]>=0.92 else None

async def retrieve_fast_async(query:str)->dict:
    cache_id=cache_key(query)
    cached=_retrieval_cache.get(cache_id)
    if cached is not None:return cached
    started=asyncio.get_running_loop().time()
    intent=fast_intent_match(query)
    if intent:
        result={"selected":[],"semantic":0,"keyword":0,"mode":"intent-fast","confidence":intent["score"],"intent":intent}
    else:
        keyword=keyword_search(query,KEYWORD_TOP_K)
        conf=keyword_confidence(query,keyword)
        if keyword and conf>=KEYWORD_FAST_MIN_OVERLAP and float(keyword[0].get("score",0.0))>=KEYWORD_FAST_MIN_BM25:
            docs=[{"text":d["text"],"metadata":d.get("metadata",{}),"_keyword_score":d.get("score",0.0)} for d in keyword[:MAX_CONTEXT_CHUNKS]]
            selected=rerank_candidates(query,docs,MAX_CONTEXT_CHUNKS)
            result={"selected":selected,"semantic":0,"keyword":len(keyword),"mode":"bm25-fast","confidence":round(conf,4)}
        else:
            semantic=await asyncio.get_running_loop().run_in_executor(None,lambda:semantic_search(query))
            fused=fuse(semantic,keyword)
            selected=rerank_candidates(query,fused,MAX_CONTEXT_CHUNKS)
            result={"selected":selected,"semantic":len(semantic),"keyword":len(keyword),"mode":"semantic-fallback","confidence":round(conf,4)}
    result["elapsed"]=asyncio.get_running_loop().time()-started
    _retrieval_cache[cache_id]=result
    if len(_retrieval_cache)>RETRIEVAL_CACHE_MAX:_retrieval_cache.pop(next(iter(_retrieval_cache)))
    return result

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
        if metadata.get("heading"):
            parts.append(str(metadata["heading"]))

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
    session_id=""
    question=""

    try:
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

        greeting=greeting_match(question)
        if greeting:
            history.append({"role":"user","content":question})
            history.append({"role":"assistant","content":greeting})
            if len(history)>HISTORY_WINDOW*2:
                session_store[session_id]=history[-(HISTORY_WINDOW*2):]
            async def greeting_stream():
                yield sse({"content":greeting,"source":"greeting","session_id":session_id})
            return StreamingResponse(greeting_stream(),media_type="text/event-stream",headers={
                "Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

        clean_question=correct_common_typos(question)
        retrieval_query=build_retrieval_query(clean_question,history)
        followup=is_followup_question(question)

        # --------------------------------------------------------
        # JSON DIRECT-ANSWER FAST PATH
        # --------------------------------------------------------
        # Only use the curated JSON response for a high-confidence,
        # non-follow-up, single-topic question. Follow-ups remain on the
        # conversational retrieval path so context is preserved.
        if not followup:
            json_hit=fast_json_match(clean_question)
            if json_hit:
                history.append({
                    "role":"user",
                    "content":question
                })
                history.append({
                    "role":"assistant",
                    "content":json_hit["response"]
                })

                if len(history)>HISTORY_WINDOW*2:
                    session_store[session_id]=history[-(HISTORY_WINDOW*2):]

                log.info(
                    "JSON direct answer | intent=%s | score=%.4f | session=%s",
                    json_hit["intent"],
                    json_hit["score"],
                    session_id
                )

                async def json_stream():
                    yield sse({
                        "content":json_hit["response"],
                        "source":"json-intent",
                        "session_id":session_id,
                        "follow_up":False,
                        "intent":json_hit["intent"],
                        "confidence":json_hit["score"],
                        "source_metadata":json_hit["source"]
                    })

                return StreamingResponse(
                    json_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control":"no-cache",
                        "Connection":"keep-alive",
                        "X-Accel-Buffering":"no"
                    }
                )

        ck=cache_key(retrieval_query)
        if ck in _response_cache:
            cached=_response_cache[ck]
            async def cached_stream():
                yield sse({"content":cached,"source":"cache","session_id":session_id,"follow_up":followup})
            return StreamingResponse(cached_stream(),media_type="text/event-stream",headers={
                "Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

        history_text="\n".join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_WINDOW:])

        retrieval_started=asyncio.get_running_loop().time()

        # Split multi-part questions only when necessary; each part gets the
        # same adaptive keyword-first/semantic-fallback strategy.
        parts=split_question_parts(retrieval_query)
        all_selected=[]
        total_semantic=0
        total_keyword=0
        modes=[]

        for part in parts[:4]:
            result=await retrieve_fast_async(part)
            all_selected.extend(result.get("selected") or [])
            total_semantic+=result.get("semantic",0)
            total_keyword+=result.get("keyword",0)
            modes.append(result.get("mode","unknown"))

        # Deduplicate strongest evidence.
        best={}
        for doc in all_selected:
            meta=doc.get("metadata") or {}
            key=str(meta.get("chunk_id") or meta.get("id") or normalize_query(doc.get("text",""))[:220])
            if key not in best or doc.get("_relevance",0)>best[key].get("_relevance",0):
                best[key]=doc
        selected=sorted(best.values(),key=lambda x:x.get("_relevance",0),reverse=True)[:MAX_CONTEXT_CHUNKS]

        retrieval_elapsed=asyncio.get_running_loop().time()-retrieval_started
        log.info(
            "Retrieval complete: %.3fs | parts=%d | semantic=%d | keyword=%d | selected=%d | modes=%s",
            retrieval_elapsed,len(parts[:4]),total_semantic,total_keyword,len(selected),','.join(modes)
        )

        context=format_context(selected,MAX_CONTEXT_CHUNKS)
        prompt_text=PROMPT.format(context=context or "No relevant context found.",history=history_text or "No previous conversation.",question=question)

    except Exception as e:
        log.exception("CHAT PIPELINE FAILED before generation | session=%s | question=%s",session_id,question[:120])
        async def pipeline_error():
            yield sse({"content":f"KALAW backend error: {type(e).__name__}: {str(e)[:400]}","source":"backend_error","session_id":session_id})
        return StreamingResponse(pipeline_error(),media_type="text/event-stream",headers={
            "Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

    async def stream():
        full=""
        llm_started=asyncio.get_running_loop().time()
        try:
            payload={"model":OLLAMA_MODEL,"prompt":prompt_text,"stream":True,"think":False,"keep_alive":"15m","options":OLLAMA_OPTIONS}
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream("POST",f"{OLLAMA_BASE_URL}/api/generate",json=payload) as response:
                    if response.status_code!=200:
                        body=await response.aread()
                        raise RuntimeError(f"Ollama HTTP {response.status_code}: {body.decode('utf-8','ignore')[:500]}")
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk=json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk["error"]))
                        token=chunk.get("response","") or ""
                        if token:
                            full+=token
                            yield sse({"content":token,"source":"qwen3","session_id":session_id,"follow_up":followup})
                        if chunk.get("done") is True:
                            break
            log.info("LLM generation complete: %.3fs | chars=%d",asyncio.get_running_loop().time()-llm_started,len(full))
            if not full.strip():
                full="Not found in the Faculty Manual."
                yield sse({"content":full,"source":"fallback","session_id":session_id,"follow_up":followup})
            history.append({"role":"user","content":question})
            history.append({"role":"assistant","content":full})
            if len(history)>HISTORY_WINDOW*2:
                session_store[session_id]=history[-(HISTORY_WINDOW*2):]
            if len(_response_cache)>=CACHE_MAX:
                _response_cache.pop(next(iter(_response_cache)))
            _response_cache[ck]=full
        except Exception as e:
            log.exception("OLLAMA GENERATION FAILED | session=%s",session_id)
            yield sse({"content":f"KALAW could not generate the answer: {type(e).__name__}: {str(e)[:400]}","source":"ollama_error","session_id":session_id,"follow_up":followup})

    return StreamingResponse(stream(),media_type="text/event-stream",headers={
        "Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"})

# ============================================================
# RETRIEVAL DIAGNOSTIC
# ============================================================

@app.get("/api/retrieval-test")
async def retrieval_test(q:str):
    q=str(q or "").strip()
    if not q:
        return {"error":"Missing query parameter: q"}

    json_hit=None
    if not is_followup_question(q):
        json_hit=fast_json_match(q)

    result=await retrieve_fast_async(q)
    selected=result.get("selected") or []

    if json_hit:
        return {
            "query":q,
            "elapsed_seconds":0.0,
            "mode":"json-intent-fast",
            "confidence":round(float(json_hit["score"]),4),
            "semantic_results":0,
            "keyword_results":0,
            "selected_results":0,
            "intent":json_hit["intent"],
            "response":json_hit["response"],
            "results":[]
        }

    return {
        "query":q,
        "elapsed_seconds":round(float(result.get("elapsed",0)),3),
        "mode":result.get("mode"),
        "confidence":result.get("confidence"),
        "semantic_results":result.get("semantic",0),
        "keyword_results":result.get("keyword",0),
        "selected_results":len(selected),
        "results":[
            {
                "relevance":round(float(d.get("_relevance",0)),4),
                "metadata":d.get("metadata") or {},
                "text":d.get("text","")[:500]
            }
            for d in selected
        ]
    }

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
        "json_fast_intents":len(_INTENT_FAST_INDEX),
        "vocabulary_words":len(_VOCABULARY),
        "chunks":len(hybrid_data["texts"]) if hybrid_data else 0,
        "cache_size":len(_response_cache),
        "embed_cache":len(_embed_cache),
        "model":"qwen3:8b",
        "embedding_model":"mxbai-embed-large:latest",
        "index_version":hybrid_data.get("version","unknown") if hybrid_data else "none",
        "ollama_base_url":OLLAMA_BASE_URL,
        "chroma_top_k":CHROMA_TOP_K,
        "keyword_top_k":KEYWORD_TOP_K,
        "max_context_chunks":MAX_CONTEXT_CHUNKS,
        "retrieval_cache_size":len(_retrieval_cache),
        "keyword_fast_overlap":KEYWORD_FAST_MIN_OVERLAP,
        "keyword_fast_bm25":KEYWORD_FAST_MIN_BM25,
        "ollama_base_url":OLLAMA_BASE_URL,
        "ollama_model":OLLAMA_MODEL
    }

# ============================================================
# RUN
# ============================================================

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="0.0.0.0",port=5000,reload=False)
