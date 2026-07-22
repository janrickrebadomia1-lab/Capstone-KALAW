import os
import shutil
import re
import pdfplumber
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
PDF_PATH    = os.path.join(DATA_DIR, "CPSU-Faculty-Manual.pdf")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Must match app.py constants so chunks are never truncated mid-retrieval
CHUNK_SIZE    = 900    # slightly under app.py MAX_CHUNK_CHARS=1000
CHUNK_OVERLAP = 100    # reduced from 150 — less redundancy, faster retrieval
MIN_CHUNK_LEN = 80     # drop fragments shorter than this

# ─────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # fix hyphenated line breaks
    text = re.sub(r"[ \t]+",      " ",    text)     # collapse spaces/tabs
    text = re.sub(r"\n{3,}",      "\n\n", text)     # collapse blank lines
    return text.strip()

# ─────────────────────────────────────────────────────────
# HEADING / ARTICLE / SECTION DETECTION  (was broken before)
# ─────────────────────────────────────────────────────────

# Matches:  "ARTICLE IV"  "ARTICLE 4"  "ARTICLE IV — Title"
_ARTICLE_RE = re.compile(
    r"^ARTICLE\s+([\w]+(?:\s+[\w]+)?)",
    re.IGNORECASE,
)

# Matches:  "SECTION 1"  "Section 2.3"  "SEC. 4"
_SECTION_RE = re.compile(
    r"^(?:SECTION|SEC\.?)\s+([\d]+(?:\.[\d]+)?)",
    re.IGNORECASE,
)

# Matches:  "CHAPTER 1"  "CHAPTER ONE"
_CHAPTER_RE = re.compile(
    r"^CHAPTER\s+(\w+)",
    re.IGNORECASE,
)

# All-caps line ≥ 10 chars that isn't a page number or table header
_CAPS_RE = re.compile(r"^[A-Z][A-Z\s,.\-]{9,}$")


def classify_line(line: str) -> tuple[str, str]:
    """
    Returns (kind, value) where kind is one of:
      'article', 'section', 'chapter', 'heading', 'body'
    """
    stripped = line.strip()

    m = _ARTICLE_RE.match(stripped)
    if m:
        return "article", m.group(1).strip()

    m = _SECTION_RE.match(stripped)
    if m:
        return "section", m.group(1).strip()

    m = _CHAPTER_RE.match(stripped)
    if m:
        return "chapter", m.group(1).strip()

    if _CAPS_RE.match(stripped) and len(stripped) >= 10:
        return "heading", stripped

    return "body", stripped

# ─────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────

def ingest_pdf():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑  Cleared old ChromaDB")

    # ── Extract text per page (keep page number for metadata) ──
    with pdfplumber.open(PDF_PATH) as pdf:
        pages = [
            (i + 1, clean_text(p.extract_text() or ""))
            for i, p in enumerate(pdf.pages)
        ]

    print(f"📄 Loaded {len(pages)} pages from PDF")

    # ── Parse into structured sections ──
    sections: list[Document] = []

    current_article = ""
    current_section = ""
    current_chapter = ""
    current_heading = "General"
    current_page    = 1
    body: list[str] = []

    def flush(page_num: int):
        """Save accumulated body lines as a Document."""
        if not body:
            return
        content = " ".join(body).strip()
        if len(content) < MIN_CHUNK_LEN:
            return

        formatted = (
            f"DOCUMENT: CPSU Faculty Manual\n"
            f"CHAPTER: {current_chapter}\n"
            f"ARTICLE: {current_article}\n"
            f"SECTION: {current_section}\n"
            f"HEADING: {current_heading}\n\n"
            f"CONTENT:\n{content}"
        )

        sections.append(Document(
            page_content=formatted,
            metadata={
                "page":    str(page_num),          # ← was missing before
                "chapter": current_chapter,
                "article": current_article,        # ← was always "" before
                "section": current_section,        # ← was always "" before
                "heading": current_heading,
                "source":  "faculty_manual",
            }
        ))
        body.clear()

    for page_num, text in pages:
        current_page = page_num

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            kind, value = classify_line(line)

            if kind == "article":
                flush(page_num)
                current_article = value
                current_section = ""          # reset section on new article
                current_heading = line.strip()

            elif kind == "section":
                flush(page_num)
                current_section = value
                current_heading = line.strip()

            elif kind == "chapter":
                flush(page_num)
                current_chapter = value
                current_article = ""
                current_section = ""
                current_heading = line.strip()

            elif kind == "heading":
                flush(page_num)
                current_heading = value

            else:  # body text
                body.append(value)

    flush(current_page)   # flush last section

    print(f"📑 Parsed {len(sections)} sections")

    # ── Split long sections into overlapping chunks ──
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(sections)

    # ── Filter + tag with chunk index ──
    final: list[Document] = []
    for i, chunk in enumerate(chunks):
        if len(chunk.page_content.strip()) < MIN_CHUNK_LEN:
            continue
        chunk.metadata["chunk_index"] = i
        final.append(chunk)

    print(f"✂️  Split into {len(final)} final chunks")

    # ── Embed + store in ChromaDB ──
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

    Chroma.from_documents(
        documents=final,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"},   # matches app.py search space
    )

    print(f"✅ Indexed {len(final)} chunks into ChromaDB at {CHROMA_PATH}")


if __name__ == "__main__":
    ingest_pdf()