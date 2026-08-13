
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

JSON_SCORE_THRESHOLD=0.90
CHROMA_TOP_K=8
CHROMA_FETCH_K=20
KEYWORD_TOP_K=8
MAX_CONTEXT_CHUNKS=6
MAX_CHUNK_CHARS=1200
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

    # Use only previous USER messages to establish the conversation subject.
    # This prevents previous assistant wording (e.g. CES compensation) from
    # contaminating a follow-up retrieval query.
    user_messages=[
        str(m.get("content","")).strip()
        for m in history
        if m.get("role")=="user" and str(m.get("content","")).strip()
    ]

    if not user_messages:
        return clean_question

    previous=user_messages[-3:]

    subject_terms=[
        "part-time","part time","full-time","full time","faculty",
        "scholar","partial scholar","full scholar","staff","employee",
        "president","mission","vision"
    ]

    subject=""
    for msg in reversed(previous):
        n=normalize_query(msg)
        if any(term in n for term in subject_terms):
            subject=msg
            break

    if not subject:
        subject=previous[-1]

    current=normalize_query(clean_question)

    # Remove follow-up pronouns; retain the actual requested topic.
    current=re.sub(
        r"\b(their|them|they|it|this|that|those|these)\b",
        " ",
        current
    )
    current=re.sub(r"\s+"," ",current).strip()

    resolved=normalize_query(f"{subject} {current}")

    log.info("Follow-up retrieval query: %s",resolved[:180])
    return resolved


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
- If the context contains direct evidence for the requested topic, never say "Not found in the Faculty Manual."\n- When a source section contains a complete list or table for the requested topic, preserve every supported item and value; do not silently omit an entry.\n- Do not use evidence from an unrelated subject or section merely because it contains the same keyword.\n"Not found in the Faculty Manual."
- Answer directly, naturally, and conversationally.
- Treat every requested topic as a separate information request.
- Answer EVERY part of a multi-part question; never answer only one topic.
- Use clear headings or bullets when there are multiple requested topics.
- Base each factual statement on the matching evidence in CONTEXT.
- Do not use a generic faculty-definition passage to answer a specific question about duties, benefits, workload, compensation, qualifications, or requirements unless it directly answers that question.
- Preserve exact numbers, requirements, conditions, dates, rates, units, and limits from the context.
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
    q=normalize_query(query)
    q_words=_expand_topics(_topic_words(query))
    if not docs:
        return []

    # Distinguish the conversational SUBJECT from the requested TOPIC.
    # The subject is a constraint; the topic determines what information
    # should be selected inside that subject.
    subject_groups={
        "part_time":{"part-time","part time","partial scholar"},
        "full_time":{"full-time","full time"},
        "faculty":{"faculty","faculty member","faculty members"},
        "scholar":{"scholar","scholarship","partial scholar","full scholar"},
        "employee":{"employee","employees","personnel","staff"},
    }

    topic_groups={
        "benefits":{"benefit","benefits","privilege","privileges","allowance",
                    "allowances","stipend","entitlement","entitlements",
                    "leave"},
        "compensation":{"compensation","salary","salaries","pay","payment",
                        "rate","rates","per","unit"},
        "teaching_load":{"teaching","load","workload","overload","units","unit"},
        "mission":{"mission"},
        "vision":{"vision"},
        "promotion":{"promotion","promotions","tenure","rank","ranking"},
        "qualification":{"qualification","qualifications","requirement",
                         "requirements","eligibility"},
        "duties":{"duty","duties","responsibility","responsibilities"},
    }

    active_subjects=[]
    for name,terms in subject_groups.items():
        if any(term in q for term in terms):
            active_subjects.append(name)

    active_topics=[]
    for name,terms in topic_groups.items():
        if any(term in q for term in terms):
            active_topics.append(name)

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

        score=(
            body_overlap*0.34+
            head_overlap*0.24+
            exact_score*0.17+
            min(float(d.get("_rrf",0.0))*10,0.05)
        )

        # Subject match is stronger than generic semantic similarity.
        subject_hits=0
        subject_misses=0
        for subject in active_subjects:
            terms=subject_groups[subject]
            hits=sum(1 for term in terms if term in text or term in meta_text)
            subject_hits += hits
            if hits == 0:
                subject_misses += 1

        # Requested topic is also explicitly rewarded.
        topic_hits=0
        for topic in active_topics:
            terms=topic_groups[topic]
            topic_hits += sum(1 for term in terms if term in text or term in meta_text)

        if active_subjects:
            score += min(0.30,subject_hits*0.10)

        if active_topics:
            score += min(0.30,topic_hits*0.08)

        # Hard-ish penalty: if the conversation subject is part-time faculty,
        # an unrelated section such as CES compensation should not outrank the
        # correct part-time compensation section merely because "compensation"
        # matches.
        if active_subjects and subject_hits == 0:
            score *= 0.28

        # If both subject and requested topic are explicit, reward the
        # intersection much more than either term alone.
        if active_subjects and active_topics and subject_hits > 0 and topic_hits > 0:
            score += 0.12

        # Remove candidates with no meaningful evidence.
        if q_words and not (q_words&body_words) and not (q_words&head_words):
            score=0.0

        item=dict(d)
        item["_relevance"]=score
        ranked.append(item)

    ranked.sort(key=lambda x:x["_relevance"],reverse=True)
    return [x for x in ranked if x["_relevance"]>0][:limit]


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
    # FOLLOW-UP-AWARE RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query=build_retrieval_query(
        clean_question,
        history
    )

    followup=is_followup_question(question)

    if followup:
        log.info(
            "Follow-up question detected | session=%s",
            session_id
        )

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

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

    selected=retrieve_relevant(retrieval_query)
    context=format_context(selected,MAX_CONTEXT_CHUNKS)

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
        "embedding_model":"mxbai-embed-large:latest"
    }

# ============================================================
# RUN
# ============================================================

if __name__=="__main__":
    import uvicorn
    uvicorn.run("app:app",host="127.0.0.1",port=5000,reload=False)

