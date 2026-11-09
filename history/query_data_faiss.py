import os, argparse, sys
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

DB_BASE = "faiss"
PROMPT = """
Aşağıdaki context’e bakarak, soruyu sadece oradaki bilgiyle yanıtlayın.
Eğer cevap yoksa: “Üzgünüm, sorduğunuz sorunun cevabı belgede yok.”

Context:
{context}

Soru: {question}
"""

def list_dbs():
    return [d for d in os.listdir(DB_BASE) if os.path.isdir(os.path.join(DB_BASE, d))]

def choose_db():
    dbs = list_dbs()
    if not dbs:
        print("❗ Index bulunamadı. populate_database.py ile indeksleyin."); sys.exit(1)
    for i, n in enumerate(dbs,1): print(f"  {i}. {n}")
    return dbs[int(input("Hangi belge? (num): "))-1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", help="Sorunuzu yazın")
    args = p.parse_args()
    q = args.question or input("📝 Sorunuz: ")
    db_name = choose_db()
    db_dir  = os.path.join(DB_BASE, db_name)

    # 1) Load FAISS
    db = FAISS.load_local(db_dir, get_embedding_function())

    # 2) Retrieval
    results = db.similarity_search_with_score(q, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc,_ in results])

    # 3) Prompt & LLM
    prompt = ChatPromptTemplate.from_template(PROMPT).format(context=context, question=q)
    answer = Ollama(model="mistral", temperature=0).invoke(prompt)

    # 4) Print
    print(f"\n▶️ Belge: {db_name}")
    print(f"🗨️ Cevap:\n{answer}\n")

if __name__=="__main__":
    main()

