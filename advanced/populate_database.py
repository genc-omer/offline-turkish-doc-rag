#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → Chroma (offline, slug klasör)
• Metin sayfaları → 350 tok +70 overlap
• Tablo satırları → ≈ 40 satırlık fact-chunk (+page meta)
"""

import argparse, shutil, warnings, unicodedata, re, math
from pathlib import Path
from typing import List

import camelot, pdfplumber, pandas as pd, tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

warnings.filterwarnings("ignore")
ENC = tiktoken.encoding_for_model("gpt2")

BASE_DIR    = Path(__file__).parent
CHROMA_BASE = BASE_DIR / "chroma"
MAX_TOKENS  = 350
OVERLAP     = 70
# Belgeye özgü üst/alt bilgi ifadelerini buraya ekleyin
UNWANTED    = {
    "Bu sayfa boş bırakılmıştır."
}

# ───────── helpers ─────────
def slugify(s: str) -> str:
    a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", a).strip("_").lower()

def strip_unwanted(txt: str) -> str:
    return "\n".join(
        ln for ln in txt.splitlines()
        if not any(u.lower() in ln.lower() for u in UNWANTED)
    ).strip()

def load_pdf(path: Path) -> List[Document]:
    docs = PyPDFLoader(str(path)).load()
    for d in docs:
        d.page_content = strip_unwanted(d.page_content)
    return docs

TOKEN_SPLITTER = TokenTextSplitter(
    encoding_name="gpt2", chunk_size=MAX_TOKENS, chunk_overlap=OVERLAP
)

# ── Tablo satırlarını ≈40’lık chunk’a dönüştür ──
def df_to_facts_dynamic(df: pd.DataFrame, tag: str, page=None) -> List[Document]:
    df = df.fillna("").astype(str)
    headers = [str(h).strip() for h in df.columns]

    rows = []
    for _, row in df.iterrows():
        hdr = str(row.iloc[0]).strip()
        if not hdr:
            continue
        cells = [str(c).strip() for c in row.iloc[1:]]
        rows.append(
            f"{hdr}: " + "; ".join(
                f"{headers[i+1]}={v}" for i, v in enumerate(cells) if v)
        )

    avg_tok = max(1, sum(len(ENC.encode(r)) for r in rows) // len(rows))
    k = max(1, min(40, math.floor(350 / avg_tok)))   # ~40 satır veya <350 tok

    docs = []
    for i in range(0, len(rows), k):
        chunk = rows[i:i+k]
        txt   = f"Tablo={tag}; Satır {i+1}-{i+len(chunk)}\n" + "\n".join(chunk)
        docs.append(Document(
            page_content=txt,
            metadata={"source": tag, "type": "table_chunk", "page": page}
        ))
    return docs

def extract_table_facts(pdf_path: Path, tag: str) -> List[Document]:
    facts: List[Document] = []
    # Camelot – çizgili
    for tbl in camelot.read_pdf(str(pdf_path), pages="all",
                                flavor="lattice", strip_text="\n", line_scale=40):
        facts += df_to_facts_dynamic(tbl.df, tag, page=int(tbl.page))
    # pdfplumber – çizgisiz
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pg, page in enumerate(pdf.pages, 1):
            for tb in page.extract_tables():
                facts += df_to_facts_dynamic(pd.DataFrame(tb), f"{tag}_p{pg}", page=pg)
    print(f"📊 {tag}: {len(facts)} tablo_fact")
    return facts

# ── Populate ──
def populate(pdf_path: Path, reset=False):
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Sadece .pdf")

    tag        = slugify(pdf_path.stem)
    chroma_dir = CHROMA_BASE / tag
    if reset and chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    chunks  = TOKEN_SPLITTER.split_documents(load_pdf(pdf_path))
    chunks += extract_table_facts(pdf_path, tag)

    db = Chroma(persist_directory=str(chroma_dir),
                embedding_function=get_embedding_function())

    texts = [d.page_content for d in chunks]
    metas = [d.metadata      for d in chunks]
    for i in range(0, len(texts), 1000):
        db.add_texts(texts=texts[i:i+1000], metadatas=metas[i:i+1000], batch_size=64)
    db.persist()
    print(f"✅ {len(chunks)} parça indekslendi → {chroma_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("pdf_path"); p.add_argument("--reset", action="store_true")
    a = p.parse_args(); populate(Path(a.pdf_path), a.reset)

