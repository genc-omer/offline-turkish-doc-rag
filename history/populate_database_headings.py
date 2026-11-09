# populate_database.py
import argparse
import os
import shutil
import re
import fitz

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma.vectorstores import Chroma  # <-- yeni

from get_embedding_function import get_embedding_function


CHROMA_BASE = "chroma"
# Belgeye özgü üst/alt bilgi ifadelerini buraya ekleyin
UNWANTED = [
    "Bu sayfa boş bırakılmıştır."
]

def clear_database(chroma_dir: str):
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

def load_and_clean(doc_path: str) -> list[Document]:
    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(doc_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(doc_path)
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {ext}")

    raw = loader.load()
    cleaned: list[Document] = []
    for doc in raw:
        lines = [
            line for line in doc.page_content.split("\n")
            if not any(tok.lower() in line.lower() for tok in UNWANTED)
        ]
        text = "\n".join(lines).strip()
        if text:
            cleaned.append(Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(doc_path),
                    "page": doc.metadata.get("page", 0)
                }
            ))
    return cleaned

def get_chunk_params(doc_path: str, docs: list[Document]) -> tuple[int, int]:
    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(doc_path)
        page_count = len(reader.pages)
        if page_count <= 30:
            return 1500, 200
        elif page_count <= 100:
            return 1000, 150
        elif page_count <= 300:
            return 800, 125
        else:
            return 600, 100
    else:
        total_words = sum(len(d.page_content.split()) for d in docs)
        avg_words   = total_words // max(len(docs), 1)
        chunk_size  = max(500, min(avg_words, 2000))
        overlap     = int(chunk_size * 0.2)
        return chunk_size, overlap

def split_by_headings(documents: list[Document]) -> list[Document]:
    pattern = re.compile(r'^(?:\d+(\.\d+)*\s+.+|[A-ZÇŞĞÜİÖ0-9 ]{5,})$', re.MULTILINE)
    sections: list[Document] = []
    for doc in documents:
        source = doc.metadata["source"]
        page   = doc.metadata["page"]
        buffer = []
        idx    = 0
        for line in doc.page_content.splitlines():
            if pattern.match(line.strip()):
                if buffer:
                    sections.append(Document(
                        page_content="\n".join(buffer).strip(),
                        metadata={**doc.metadata, "id": f"{source}:{page}:sec{idx}"}
                    ))
                    idx += 1
                buffer = [line]
            else:
                buffer.append(line)
        if buffer:
            sections.append(Document(
                page_content="\n".join(buffer).strip(),
                metadata={**doc.metadata, "id": f"{source}:{page}:sec{idx}"}
            ))
    return sections

def split_documents_dynamic(doc_path: str, documents: list[Document]) -> list[Document]:
    # 1) Renk bazlı heading split
    color_chunks = split_by_colored_headings(doc_path, documents)
    if len(color_chunks) > len(documents):
        print("ℹ️ Color‑based heading splitting uygulandı.")
        return color_chunks

    # 2) Eski başlık‑bazlı regex split
    heading_chunks = split_by_headings(documents)
    if len(heading_chunks) > len(documents):
        print("ℹ️ Regex‑based heading splitting uygulandı.")
        return heading_chunks

    # 3) Fallback: size‑based splitting
    chunk_size, chunk_overlap = get_chunk_params(doc_path, documents)
    print(f"⚙️ chunk_size={chunk_size}, overlap={chunk_overlap} (dynamic fallback)\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(documents)

def split_by_colored_headings(doc_path: str, documents: list[Document]) -> list[Document]:
    """
    PyMuPDF ile PDF içindeki her span'ın rengini kontrol eder.
    Siyah (0) dışındaki renklerdeki span'ları heading kabul edip
    sayfa metnini o başlıklara göre böler.
    """
    # 1) Önce PyMuPDF ile tüm başlık metinlerini çıkaralım
    doc = fitz.open(doc_path)
    colored_headings = {}  # { (page, text) }
    for p in range(len(doc)):
        page = doc[p]
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:  # yalnızca text
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    # span["color"] sıfırsa siyah demektir
                    if span["color"] != 0:
                        txt = span["text"].strip()
                        if txt:
                            colored_headings.setdefault(p, set()).add(re.escape(txt))

    # 2) Renkli başlıklara göre split
    sections: list[Document] = []
    for doc_obj in documents:
        source = doc_obj.metadata["source"]
        page   = doc_obj.metadata["page"]
        text   = doc_obj.page_content
        patterns = colored_headings.get(page, set())
        if not patterns:
            # o sayfada renkli başlık yoksa, tek parça döndür
            sections.append(doc_obj)
            continue

        # tek bir regex ile tüm renkli başlıkları yakala
        header_regex = re.compile(r"(?m)^(" + "|".join(patterns) + r")\s*$")
        buffer = []
        idx    = 0
        for line in text.splitlines():
            if header_regex.match(line):
                # önceki buffer'ı flush et
                if buffer:
                    sections.append(Document(
                        page_content="\n".join(buffer).strip(),
                        metadata={**doc_obj.metadata, "id": f"{source}:{page}:colsec{idx}"}
                    ))
                    idx += 1
                buffer = [line]
            else:
                buffer.append(line)
        # son buffer
        if buffer:
            sections.append(Document(
                page_content="\n".join(buffer).strip(),
                metadata={**doc_obj.metadata, "id": f"{source}:{page}:colsec{idx}"}
            ))

    return sections

def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_source_page = None
    idx = 0
    for chunk in chunks:
        src  = chunk.metadata["source"]
        pg   = chunk.metadata["page"]
        key  = f"{src}:{pg}"
        if key == last_source_page:
            idx += 1
        else:
            idx = 0
        chunk.metadata["id"] = f"{key}:{idx}"
        last_source_page = key
    return chunks

def populate(doc_path: str, reset: bool = False):
    base_name  = os.path.splitext(os.path.basename(doc_path))[0]
    chroma_dir = os.path.join(CHROMA_BASE, base_name)
    if reset:
        clear_database(chroma_dir)

    # 1) Load & clean
    docs = load_and_clean(doc_path)

    # 2) Split
    chunks = split_documents_dynamic(doc_path, docs)

    # 3) ID’leri ayarla
    chunks = calculate_chunk_ids(chunks)

    # Yeni Chroma API’si ile sadece bu iki parametre yeterli:
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=get_embedding_function(),
    )

    db.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])

    print(f"✅ Toplam {len(chunks)} chunk, `{chroma_dir}` klasörüne kaydedildi.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="İndekslenecek .pdf veya .docx dosya yolu")
    parser.add_argument("--reset", action="store_true",
                        help="Mevcut dizini temizle ve yeniden oluştur")
    args = parser.parse_args()
    populate(args.doc_path, args.reset)

