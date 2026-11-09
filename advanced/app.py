#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit PDF QA – Eksiksizlik odaklı
"""

import streamlit as st, hashlib, re, unicodedata, tiktoken, time, logging
from pathlib import Path
from typing import List, Set
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function

# ─── Konfig ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma"
RERANK     = "./models/msmarco-ce"
ANN_STEP, ANN_MAX = 30, 180
SCORE_TH, OVL_TH  = 0.20, 0.30
MAX_CTX    = 2_000
ENC        = tiktoken.encoding_for_model("gpt2")

ce         = CrossEncoder(RERANK, device="cpu")
llm        = Ollama(model="mistral", temperature=0, repeat_penalty=1.3, num_predict=0)

# ─── Yardımcı fonksiyonlar ──────────────────────────────
slug = lambda s: re.sub(r"[^A-Za-z0-9_-]+","_", unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode()).strip("_").lower()
tok  = lambda t: set(re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğışöü]+", t.lower()))
uid  = lambda d: d.metadata.get("id") or hashlib.md5(d.page_content.encode()).hexdigest()

@st.cache_resource
def load_db(coll):                                   # Chroma cache
    return Chroma(persist_directory=str(CHROMA_DIR/coll),
                  embedding_function=get_embedding_function())

def merge_rows(docs: List[Document]) -> List[Document]:
    merged, last = [], None
    for d in docs:
        if last and d.metadata.get("source")==last.metadata.get("source") \
           and "Satır" in d.page_content and "Satır" in last.page_content:
            try:
                pe=int(last.page_content.split("Satır ")[1].split("-")[1].split("\n")[0])
                ts=int(d.page_content.split("Satır ")[1].split("-")[0])
                if ts==pe+1:
                    last.page_content+="\n"+d.page_content.split("\n",1)[1]
                    continue
            except: pass
        merged.append(d); last=d
    return merged

def get_needed_items(question:str)->Set[str]:
    prompt=f"Soru: {question}\nYanıt için gerekebilecek anahtar kelime veya tablo başlıklarını virgülle sırala."
    items=llm.invoke(prompt)
    return {i.strip().lower() for i in items.split(",") if i.strip()}

def build_context(question:str, db):
    needed = get_needed_items(question)
    covered=set(); ctx=[]; tot_tok=0; seen=set()
    for k in range(ANN_STEP, ANN_MAX+ANN_STEP, ANN_STEP):
        batch=[(d,s) for d,s in db.similarity_search_with_score(question,k=k) if uid(d) not in seen]
        if not batch: continue
        seen|={uid(d) for d,_ in batch}
        scores = ce.predict([(question,d.page_content) for d,_ in batch])
        ranked=[d for ((d,_),sc) in sorted(zip(batch,scores), key=lambda z:z[1], reverse=True)
                 if sc>=SCORE_TH and len(tok(d.page_content)&tok(question))/len(tok(question))>=OVL_TH]
        for d in merge_rows(ranked):
            t=len(ENC.encode(d.page_content))
            if tot_tok+t>MAX_CTX: break
            ctx.append(d); tot_tok+=t
            covered |= tok(d.page_content)
        if needed.issubset(covered) or tot_tok>=MAX_CTX:
            break
    return "\n\n---\n\n".join(d.page_content for d in ctx), ctx, needed

def self_check(answer:str, needed:Set[str])->Set[str]:
    prompt=f"GEREKENLER: {', '.join(sorted(needed))}\nYANIT: {answer}\nEksik öğe varsa listele, yoksa 'tam'."
    resp=llm.invoke(prompt).strip().lower()
    if resp=="tam":
        return set()
    return {w.strip().lower() for w in resp.split(",") if w.strip()}

def majority_merge(ans_list:List[str])->str:
    if len(set(ans_list))==1: return ans_list[0]
    prompt="Bu üç cevabı birleştir, hiçbir bilgi eksik kalmasın:\n\n"+"\n\n###\n\n".join(ans_list)
    return llm.invoke(prompt)

# ─── Streamlit UI ───────────────────────────────────────
st.set_page_config(page_title="PDF QA (eksiksiz)", page_icon="📑")
st.title("📑 Tek-Belge Soru-Cevap (Eksiksiz)")

colls=[d.name for d in CHROMA_DIR.iterdir() if d.is_dir()]
sel  = st.selectbox("Belge seç", options=colls)
question = st.text_input("Soru yazın ve Enter’a basın")

if question and sel:
    db   = load_db(sel)

    # 1) Bağlam oluştur
    t0=time.perf_counter()
    context, docs, needed = build_context(question, db)
    st.caption(f"🔍 Retrieval: { (time.perf_counter()-t0)*1e3:.0f} ms, chunks={len(docs)}")

    with st.expander("Bağlam parçaları"):
        for d in docs:
            pg=d.metadata.get("page","?")
            st.write(f"**s={pg}** — {d.page_content.splitlines()[0][:120]}…")

    if not context:
        st.warning("Üzgünüm, sorduğunuz sorunun cevabı belgede yok.")
    else:
        # 2) İlk cevap + self-check
        prompt=f"""Aşağıdaki “context” bölümündeki bilgiler dışına çıkmadan Türkçe yanıt ver.
Tablo satırlarını kopyalama; bilgiyi cümle içinde kullan.

Context:
{context}

Soru: {question}"""
        answer = llm.invoke(prompt)
        missing = self_check(answer, needed)

        if missing:
            st.info("🔄 Eksik öğe tespit edildi, bağlam genişletiliyor…")
            # ∎ Eksik kelimeleri soruya ekle (OR) ve tekrar bağlam kur
            new_q = question + " " + " ".join(missing)
            context, docs, _ = build_context(new_q, db)
            answer = llm.invoke(prompt.replace(context, context))  # güncellenmiş bağlam

        # 3) Çoklu çıktı uzlaşması
        ans_list=[llm.invoke(prompt, seed=s) for s in (1,2,3)]
        final_answer = majority_merge(ans_list)

        st.success(final_answer)

