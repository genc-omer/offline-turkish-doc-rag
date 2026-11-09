# query_rag.py
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
import os, argparse, sys
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function



warnings.filterwarnings("ignore", category=DeprecationWarning)
CHROMA_BASE = "chroma"
PROMPT_TEMPLATE = """
Aşağıdaki “context” bölümündeki bilgiler dışına kesinlikle çıkmadan, gelen soruyu yalnızca oradaki veriye dayanarak Türkçe yanıtlayın.
Eğer sorunun cevabı context’te yoksa, yalnızca şu cümleyi yazın:
“Üzgünüm, sorduğunuz sorunun cevabı belgede yok.”

Context:
{context}

Soru: {question}
"""

"""
CHROMA_BASE klasöründe bulunan tüm indekslenmiş belgelerin listesini döndürür.

Input: None
Output: list[str] -> Belgelerin isimlerini (klasör adlarını) içeren liste.
"""
def list_available_pdfs():
    return [d for d in os.listdir(CHROMA_BASE) if os.path.isdir(os.path.join(CHROMA_BASE, d))]


"""
Kullanıcıya mevcut indekslenmiş belgeleri listeler, seçim yaptırır ve seçilen belge adını döndürür. Eğer hiç belge yoksa programı sonlandırır.

Input: None
Output: str -> Kullanıcının seçtiği belgenin ismi.
"""
def choose_pdf():
    pdfs = list_available_pdfs()
    if not pdfs:
        print("❗ Hiç PDF bulunamadı. Lütfen önce populate_database.py ile indeksleyin.")
        sys.exit(1)
    print("📚 Mevcut belgeler:")
    for i, name in enumerate(pdfs, 1):
        print(f"  {i}. {name}")
    choice = int(input("Hangi belgeyi kullanmak istersiniz? (numara): "))
    return pdfs[choice - 1]

"""
Kullanıcıdan soru alır, belge seçtirir, Chroma veritabanından benzer chunk'ları çeker, Ollama LLM kullanarak yanıt üretir ve çıktıyı ekrana yazdırır.

Input: None (komut satırından opsiyonel olarak "question" parametresi alabilir)
Output: None (cevap konsola yazdırılır)
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Sorunuzu yazın")
    args = parser.parse_args()
    question = args.question or input("📝 Sorunuz: ")
    pdf_name = choose_pdf()
    chroma_dir = os.path.join(CHROMA_BASE, pdf_name)

    # 1) DB’yi yükle
    db = Chroma(persist_directory=chroma_dir, embedding_function=get_embedding_function())
    # 2) Top‑k benzerlik
    results = db.similarity_search_with_score(question, k=5)

    # 3) Context oluştur
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=question)

    # 4) Sorgula
    llm = Ollama(model="mistral", temperature=0)
    answer = llm.invoke(prompt)

    # 5) Çıktı
    print(f"\n▶️ Belge: {pdf_name}")
    print(f"🗨️ Cevap:\n{answer}\n")
    '''
        print("📑 Kaynak chunk ID’leri ve içerikleri:\n")
        for doc, score in results:
            cid = doc.metadata.get("id")
            similarity = (2.0 - score) * 100
            print(f"🔹 Chunk ID: {cid} → {similarity:.2f}% benzerlik")
            print("    İçerik:")
            for line in doc.page_content.strip().split("\n"):
                print(f"      {line}")
            print("\n" + ("─" * 30) + "\n")
    '''

if __name__ == "__main__":
    main()

