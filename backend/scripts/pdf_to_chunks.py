import os, shutil, re, pdfplumber
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ["ANONYMIZED_TELEMETRY"]="False"
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,"data")
PDF_PATH=os.path.join(DATA_DIR,"CPSU-Faculty-Manual.pdf")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")
CHUNK_SIZE=900
CHUNK_OVERLAP=100
MIN_CHUNK_LEN=80

def clean_text(text:str)->str:
    text=text.replace("ﬁ","fi").replace("ﬂ","fl").replace("ﬀ","ff")
    text=re.sub(r"(\w)-\n(\w)",r"\1\2",text)
    text=re.sub(r"[ \t]+"," ",text)
    text=re.sub(r"\n{3,}","\n\n",text)
    return text.strip()

ARTICLE_RE=re.compile(r"^ARTICLE\s+([\w]+(?:\s+[\w]+)?)",re.I)
SECTION_RE=re.compile(r"^(?:SECTION|SEC\.?)\s+([\d]+(?:\.[\d]+)?)",re.I)
CHAPTER_RE=re.compile(r"^CHAPTER\s+(\w+)",re.I)
CAPS_RE=re.compile(r"^[A-Z][A-Z\s,.\-]{9,}$")

def classify_line(line:str)->tuple[str,str]:
    s=line.strip()
    m=ARTICLE_RE.match(s)
    if m:return "article",m.group(1).strip()
    m=SECTION_RE.match(s)
    if m:return "section",m.group(1).strip()
    m=CHAPTER_RE.match(s)
    if m:return "chapter",m.group(1).strip()
    if CAPS_RE.match(s) and len(s)>=10:return "heading",s
    return "body",s

def ingest_pdf():
    if not os.path.exists(PDF_PATH):raise FileNotFoundError(f"Faculty Manual PDF not found: {PDF_PATH}")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old ChromaDB")
    with pdfplumber.open(PDF_PATH) as pdf:
        pages=[(i+1,clean_text(p.extract_text() or "")) for i,p in enumerate(pdf.pages)]
    print(f"Loaded {len(pages)} pages from PDF")
    sections=[]; current_article=""; current_section=""; current_chapter=""; current_heading="General"; body=[]
    def flush(page_num:int):
        if not body:return
        content=" ".join(body).strip()
        if len(content)<MIN_CHUNK_LEN:return
        formatted=(f"DOCUMENT: CPSU Faculty Manual\nCHAPTER: {current_chapter}\nARTICLE: {current_article}\nSECTION: {current_section}\nHEADING: {current_heading}\n\nCONTENT:\n{content}")
        sections.append(Document(page_content=formatted,metadata={"page":str(page_num),"chapter":current_chapter,"article":current_article,"section":current_section,"heading":current_heading,"source":"faculty_manual"}))
        body.clear()
    for page_num,text in pages:
        for raw_line in text.split("\n"):
            line=raw_line.strip()
            if not line:continue
            kind,value=classify_line(line)
            if kind=="article":
                flush(page_num); current_article=value; current_section=""; current_heading=line
            elif kind=="section":
                flush(page_num); current_section=value; current_heading=line
            elif kind=="chapter":
                flush(page_num); current_chapter=value; current_article=""; current_section=""; current_heading=line
            elif kind=="heading":
                flush(page_num); current_heading=value
            else:body.append(value)
    flush(pages[-1][0] if pages else 1)
    print(f"Parsed {len(sections)} sections")
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP,separators=["\n\n","\n",". "," ",""])
    chunks=splitter.split_documents(sections)
    final=[]
    for chunk in chunks:
        if len(chunk.page_content.strip())<MIN_CHUNK_LEN:continue
        chunk.metadata["chunk_index"]=len(final)
        final.append(chunk)
    print(f"Split into {len(final)} final chunks")
    embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest")
    Chroma.from_documents(documents=final,embedding=embeddings,persist_directory=CHROMA_PATH,collection_metadata={"hnsw:space":"cosine"})
    print(f"Indexed {len(final)} chunks into ChromaDB at {CHROMA_PATH}")

if __name__=="__main__":ingest_pdf()
