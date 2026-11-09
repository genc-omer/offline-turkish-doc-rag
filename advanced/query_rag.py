#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genişleyen ANN 30→… ; Kapsam döngüsü
Bağlam tüm soru kelimelerini içerene dek parça ekler
Bağlam ≤ 2 000 token – Ardışık tablo satırları birleştirilir
"""

import argparse, sys, warnings, re, unicodedata, hashlib, tiktoken
from pathlib import Path
from typing import List, Tuple, Set

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function

warnings.filterwarnings("ignore")
ENC = tiktoken.encoding_for_model("gpt2")

# ——— Ayarlar ————————————————————————————
BASE_DIR       = Path(__file__).parent
CHROMA_BASE    = BASE_DIR / "chroma"
RERANK_PATH    = "./models/msmarco-ce"
ANN_STEP       = 30
ANN_MAX        = 180
SCORE_TH       = 0.20
OVL_TH         = 0.30
MAX_CTX_TOKENS = 2_000
reranker       = CrossEncoder(RERANK_PATH, device="cpu")

# ——— Yardımcılar ——————————————————————————
def slugify(s:str) -> str:
    a = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
    return re.sub(r"[^A-Za-z0-9_-]+","_",a).strip("_").lower()

def list_colls() -> List[str]:
    return sorted([d.name for d in CHROMA_BASE.iterdir() if d.is_dir()])

def choose_coll() -> str:
    colls = list_colls()
    if not colls:
        sys.exit("❗ Önce populate işlemi yapın.")
    for i,n in enumerate(colls,1): print(f"  {i}. {n}")
    return colls[int(input("Hangi belge? (numara): "))-1]

def tokens(txt:str) -> Set[str]:
    return set(re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğışöü]+", txt.lower()))

def uniq_id(doc:Document) -> str:
    """id metadatası yoksa içerik-hash yedek.”"""
    return doc.metadata.get("id") or hashlib.md5(doc.page_content.encode()).hexdigest()

def merge_rows(docs:List[Document]) -> List[Document]:
    merged,last=[],None
    for d in docs:
        if (last and d.metadata.get("source")==last.metadata.get("source")
            and "Satır" in d.page_content and "Satır" in last.page_content):
            try:
                pe=int(last.page_content.split("Satır ")[1].split("-")[1].split("\n")[0])
                ts=int(d.page_content.split("Satır ")[1].split("-")[0])
                if ts==pe+1:
                    last.page_content+="\n"+d.page_content.split("\n",1)[1]; continue
            except Exception: pass
        merged.append(d); last=d
    return merged

# ——— Retrieval + kapsam döngüsü ——————————
def retrieve_with_coverage(question:str, db) -> Tuple[str,List[Document]]:
    q_set=tokens(question); seen=set(); ctx_docs=[]; tok=0
    for k in range(ANN_STEP, ANN_MAX+ANN_STEP, ANN_STEP):
        batch=db.similarity_search_with_score(question,k=k)
        # yeni & benzersiz
        batch=[(d,s) for d,s in batch if uniq_id(d) not in seen]
        if not batch: break
        seen |= {uniq_id(d) for d,_ in batch}

        scores=reranker.predict([(question,d.page_content) for d,_ in batch])
        ranked=[d for ((d,_),sc) in sorted(zip(batch,scores),key=lambda z:z[1],reverse=True)
                if sc>=SCORE_TH and len(tokens(d.page_content)&q_set)/len(q_set)>=OVL_TH]

        for d in merge_rows(ranked):
            t=len(ENC.encode(d.page_content))
            if tok+t>MAX_CTX_TOKENS: break
            ctx_docs.append(d); tok+=t
            q_set-=tokens(d.page_content)
        if not q_set or tok>=MAX_CTX_TOKENS: break

    context="\n\n---\n\n".join(d.page_content for d in ctx_docs)
    return context, ctx_docs

# ——— Ana akış ————————————————————————————
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("question",nargs="?"); a=ap.parse_args()
    question=a.question or input("📝 Soru: ")
    coll=choose_coll()

    db=Chroma(persist_directory=str(CHROMA_BASE/coll),
              embedding_function=get_embedding_function())

    context,docs=retrieve_with_coverage(question,db)

    print("\n—— Bağlam listesi ——")
    for d in docs:
        pg=d.metadata.get("page","?")
        print(f"s={pg:<3}  {d.page_content.splitlines()[0][:100]}…")
    print("—— son ——\n")

    if not context.strip():
        print("🗨️ Üzgünüm, sorduğunuz sorunun cevabı belgede yok."); return

    prompt=f"""Aşağıdaki “context” bölümündeki bilgiler dışına çıkmadan Türkçe yanıt ver.
Tablo satırlarını kopyalama; bilgiyi cümle içinde kullan.
Eğer cevap yoksa şu cümleyi yaz: "Üzgünüm, sorduğunuz sorunun cevabı belgede yok."

Context:
{context}

Soru: {question}
"""
    ans=Ollama(model="mistral",temperature=0,repeat_penalty=1.3,num_predict=0).invoke(prompt)
    print("🗨️ Cevap:\n"+ans+"\n")

if __name__=="__main__":
    main()

