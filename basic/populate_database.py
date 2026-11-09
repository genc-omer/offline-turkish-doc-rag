# populate_database.py
import argparse, os
import shutil

from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_BASE = "chroma"
# Belgeye özgü üst/alt bilgi ifadelerini buraya ekleyin
UNWANTED = [
    "Bu sayfa boş bırakılmıştır."
]

"""
Mevcut Chroma veritabanı klasörünü tamamen siler.

Input:chroma_dir (str): Silinecek veritabanın klasör yolu.
Output:None
"""
def clear_database(chroma_dir: str):
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)

"""
PDF veya DOCX dosyasını yükler ve UNWANTED listesindeki ifadeleri temizler.

Input: doc_path (str): Dosya yolu.
Output: list[Document]: Temizlenmiş belgeler listesi.
"""
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


"""
Chunk boyutu ve overlap değerlerini dinamik olarak belirler.
PDF için sayfa sayısına, DOCX için kelime sayısına göre hesaplama yapar.

Input: doc_path (str): Dosya yolu, docs (list[Document]): Temizlenmiş belge listesi.
Output: tuple[int, int]: (chunk_size, chunk_overlap) değerleri.
"""
def get_chunk_params(doc_path: str, docs: list[Document]) -> tuple[int, int]:
    ext = os.path.splitext(doc_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(doc_path)
        page_count = len(reader.pages)
        # PDF için kullandığımız eşikler:
        if page_count <= 30:
            return 1500, 200
        elif page_count <= 100:
            return 1000, 150
        elif page_count <= 300:
            return 800, 125
        else:
            return 600, 100
    else:
        # Word için kelime sayısına göre dinamik:
        total_words = sum(len(d.page_content.split()) for d in docs)
        # Ortalama kelime sayısı / doc:
        avg_words = total_words // max(len(docs), 1)
        # Min 500, max 2000 arasında:
        chunk_size = max(500, min(avg_words, 2000))
        overlap    = int(chunk_size * 0.2)
        return chunk_size, overlap

"""
Belgeyi dinamik chunk_size ve chunk_overlap ile parçalara böler.

Input:doc_path (str): Dosya yolu, docs (list[Document]): Temizlenmiş belge listesi.
Output:list[Document]: Parçalanmış belge listesi.
"""
def split_documents_dynamic(doc_path: str, docs: list[Document]) -> list[Document]:
    chunk_size, chunk_overlap = get_chunk_params(doc_path, docs)
    print(f"📄 Kaynak: {os.path.basename(doc_path)}")
    print(f"   chunk_size={chunk_size}, overlap={chunk_overlap}\n")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    return splitter.split_documents(docs)

"""
Her chunk’a benzersiz bir ID ekler: {dosya}:{sayfa}:{index}.

Input:chunks (list[Document]): Chunk’lanmış belgeler listesi.
Output:list[Document]: ID eklenmiş belgeler listesi.
"""
def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_page = None
    idx = 0
    for chunk in chunks:
        src = chunk.metadata["source"]
        pg  = chunk.metadata["page"]
        current = f"{src}:{pg}"
        if current == last_page:
            idx += 1
        else:
            idx = 0
        chunk.metadata["id"] = f"{current}:{idx}"
        last_page = current
    return chunks

"""
Dosyayı yükler, temizler, chunk’lara böler, ID atar ve Chroma veritabanına kaydeder.

Input:doc_path (str): PDF veya DOCX dosya yolu, reset (bool): True ise mevcut veritabanı silinir.
Output:None
"""
def populate(doc_path: str, reset: bool = False):
    pdf_name   = os.path.splitext(os.path.basename(doc_path))[0]
    chroma_dir = os.path.join(CHROMA_BASE, pdf_name)
    if reset:
        clear_database(chroma_dir)

    docs   = load_and_clean(doc_path)
    chunks = split_documents_dynamic(doc_path, docs)
    chunks = calculate_chunk_ids(chunks)

    print("🔍 Oluşan Chunk’lar:\n")
    for chunk in chunks:
        cid = chunk.metadata["id"]
        print(f"🔹 Chunk ID: {cid}")
        for line in chunk.page_content.strip().split("\n"):
            print(f"    {line}")
        print("\n" + ("─" * 40) + "\n")

    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=get_embedding_function()
    )
    db.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])
    db.persist()

    print(f"✅ Toplam {len(chunks)} chunk, `{chroma_dir}` dizinine eklendi.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("doc_path", help="İndekslenecek .pdf veya .docx dosyası")
    parser.add_argument(
        "--reset", action="store_true",
        help="Mevcut chroma klasörünü silip yeniden oluştur"
    )
    args = parser.parse_args()
    populate(args.doc_path, args.reset)
