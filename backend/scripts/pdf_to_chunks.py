
import os,shutil,re,pdfplumber
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

os.environ["ANONYMIZED_TELEMETRY"]="False"

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR=os.path.join(BASE_DIR,"data")
PDF_PATH=os.path.join(DATA_DIR,"CPSU-Faculty-Manual.pdf")
CHROMA_PATH=os.path.join(DATA_DIR,"chroma_db")

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
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

def classify_line(line:str):
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
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Faculty Manual PDF not found: {PDF_PATH}")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared old ChromaDB.")

    with pdfplumber.open(PDF_PATH) as pdf:
        pages=[(i+1,clean_text(p.extract_text() or "")) for i,p in enumerate(pdf.pages)]

    print(f"Loaded {len(pages)} pages from PDF.")

    sections=[]
    current_chapter=""
    current_article=""
    current_section=""
    current_heading="General"
    body=[]
    current_page=1

    def flush(page_num:int):
        nonlocal body,current_page
        if not body:return

        content=" ".join(body).strip()

        if len(content)<MIN_CHUNK_LEN:
            body=[]
            return

        formatted=(
            "DOCUMENT: CPSU Faculty Manual\n"
            f"CHAPTER: {current_chapter}\n"
            f"ARTICLE: {current_article}\n"
            f"SECTION: {current_section}\n"
            f"HEADING: {current_heading}\n\n"
            f"CONTENT:\n{content}"
        )

        sections.append(
            Document(
                page_content=formatted,
                metadata={
                    "page":str(page_num),
                    "chapter":current_chapter,
                    "article":current_article,
                    "section":current_section,
                    "heading":current_heading,
                    "source":"faculty_manual"
                }
            )
        )

        body=[]
        current_page=page_num

    for page_num,text in pages:
        current_page=page_num

        for raw_line in text.split("\n"):
            line=raw_line.strip()

            if not line:
                continue

            kind,value=classify_line(line)

            if kind=="chapter":
                flush(page_num)
                current_chapter=value
                current_article=""
                current_section=""
                current_heading=line

            elif kind=="article":
                flush(page_num)
                current_article=value
                current_section=""
                current_heading=line

            elif kind=="section":
                flush(page_num)
                current_section=value
                current_heading=line

            elif kind=="heading":
                flush(page_num)
                current_heading=value

            else:
                body.append(value)

        if body:
            body.append("")

    flush(current_page)

    print(f"Parsed {len(sections)} logical sections.")

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n","\n",". "," ",""]
    )

    chunks=splitter.split_documents(sections)

    final=[]

    for chunk in chunks:
        text=chunk.page_content.strip()

        if len(text)<MIN_CHUNK_LEN:
            continue

        chunk.metadata["chunk_index"]=len(final)
        chunk.metadata["chunk_length"]=len(text)

        final.append(chunk)

    print(f"Created {len(final)} searchable chunks.")

    embeddings=OllamaEmbeddings(
        model="mxbai-embed-large:latest"
    )

    db=Chroma.from_documents(
        documents=final,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("ChromaDB successfully created.")
    print(f"Chunks stored: {len(final)}")
    print(f"Database: {CHROMA_PATH}")

if __name__=="__main__":
    ingest_pdf()

