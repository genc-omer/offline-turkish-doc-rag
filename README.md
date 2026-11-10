# Offline Turkish Document QA (RAG)

A retrieval-augmented generation (RAG) system for asking questions about Turkish technical documents. Everything runs locally: the embedding model, the reranker and the LLM. No external API is called at query time.

The project has two versions, built in July 2025:

- **basic**: a straightforward pipeline for PDF and DOCX files.
- **advanced**: adds table extraction, cross-encoder reranking, a coverage-driven retrieval loop and a Streamlit app.

## The two versions

| | basic | advanced |
|---|---|---|
| Input | PDF, DOCX | PDF |
| Chunking | Character-based; chunk size chosen by document length | 350 tokens with 70 overlap, plus separate table chunks |
| Tables | Indexed as plain page text | Extracted with Camelot and pdfplumber; each row becomes a `header: column=value` fact |
| Retrieval | Top-5 similarity search | Widening search (30 → 180 candidates), cross-encoder reranking, keyword-coverage loop, 2,000-token context cap |
| Interface | CLI | CLI and Streamlit app |

Both versions use [LaBSE](https://huggingface.co/sentence-transformers/LaBSE) embeddings, keep a separate Chroma collection per document, and answer with Mistral 7B through Ollama. Prompts instruct the model to answer in Turkish using only the retrieved context. The CLI scripts also tell it to reply with a fixed "not in the document" sentence when the answer is missing.

### basic

1. **Load and clean.** Reads a PDF or DOCX and removes repeated header/footer lines listed in `UNWANTED`.
2. **Chunk.** For PDFs, the chunk size depends on page count: up to 30 pages → 1,500 characters (200 overlap); up to 100 → 1,000 (150); up to 300 → 800 (125); longer → 600 (100). For DOCX, the size is derived from the average word count per section, clamped to 500–2,000, with 20% overlap.
3. **Index.** Gives each chunk an ID of the form `file:page:index` and stores it in `chroma/<file name>/`.
4. **Query.** Lists the indexed documents, retrieves the 5 closest chunks and asks Mistral.

### advanced

**Ingestion** (`populate_database.py`):
- **Text:** split into token-based chunks of 350 tokens with 70 overlap.
- **Tables:** extracted with Camelot (lattice) and pdfplumber. Each row is rewritten as `Row header: column=value; column=value`. Rows are grouped into chunks of up to 40 rows or about 350 tokens, each prefixed with `Tablo=<document>; Satır i-j`.
- **Storage:** stored in `advanced/chroma/<document slug>/`.

**Query** (`query_rag.py`):
1. Retrieve 30 candidates, widening in steps of 30 up to 180.
2. Rerank with a cross-encoder. Drop chunks that score below 0.20 or that contain fewer than 30% of the question's words.
3. Add chunks until every question word is covered or the context reaches 2,000 tokens.
4. Merge consecutive table-row chunks.
5. Ask Mistral, instructing it to use table data in sentences instead of copying rows.

**App** (`app.py`) is a Streamlit version of the query step with three additions:
- The LLM first lists the items needed to answer the question.
- The LLM checks its answer for missing items.
- Three answers are generated and merged.

## Repository layout

```
basic/
  get_embedding_function.py   LaBSE, loaded from models/labse on CPU
  populate_database.py        ingestion (PDF/DOCX)
  query_data.py               CLI question answering
advanced/
  get_embedding_function.py   same as basic
  populate_database.py        ingestion with table extraction
  query_rag.py                CLI question answering with reranking
  app.py                      Streamlit app
history/                      intermediate experiments, kept for reference
requirements.txt
```

## Setup

### 1. Python environment

Use Python 3.12. On Python 3.11 and below, the pinned versions conflict (camelot-py 1.0.0 requires `pypdf<4` there).

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Windows has no `python3.12` command. There, create the environment with conda instead:

```bash
conda create -n turkish-rag python=3.12
conda activate turkish-rag
pip install -r requirements.txt
```

`requirements.txt` pins the main packages to the versions current in late July 2025. To also pin every transitive dependency to that date, install with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -r requirements.txt --exclude-newer 2025-08-01
```

### 2. Local models

The code loads models from `models/`, relative to the directory you run it from. Run everything from the repository root.

| Path | Used by | Model |
|---|---|---|
| `models/labse` | both versions | `sentence-transformers/LaBSE` (about 1.9 GB) |
| `models/msmarco-ce` | advanced only | an MS MARCO cross-encoder (about 0.5 GB; see note) |

```bash
python -c "from huggingface_hub import snapshot_download as d; skip=['*.h5','*.msgpack','*.bin','onnx/*','openvino/*']; d('sentence-transformers/LaBSE', local_dir='models/labse', ignore_patterns=skip); d('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', local_dir='models/msmarco-ce', ignore_patterns=skip)"
```

The `skip` patterns leave out the TensorFlow, Flax, ONNX, OpenVINO and PyTorch `.bin` copies of the weights; the code loads the safetensors files.

Then switch the reranker to sigmoid scores: the 0.20 threshold in `query_rag.py` and `app.py` expects 0–1 scores, and the mMARCO model outputs raw logits.

```bash
python -c "import json, pathlib; p=pathlib.Path('models/msmarco-ce/config.json'); c=json.loads(p.read_text()); c['sbert_ce_default_activation_function']='torch.nn.modules.activation.Sigmoid'; p.write_text(json.dumps(c, indent=2))"
```

> **Reranker note:** the exact cross-encoder checkpoint used originally was not recorded. The multilingual mMARCO model above is the suggested replacement for Turkish text. With sigmoid scores it is on the same 0–1 scale as the 0.20 threshold, but that threshold was tuned for the original model, so it may still need adjusting.

### 3. Ollama

Install [Ollama](https://ollama.com), then pull Mistral 7B Instruct v0.3 and register it under the name the code uses:

```bash
ollama pull mistral:v0.3
ollama cp mistral:v0.3 mistral
```

The code requests a model called `mistral`. Copying the `v0.3` tag under that name keeps the July 2025 model without changing any code.

### 4. Tokenizer files (advanced only)

The advanced version counts tokens with tiktoken's GPT-2 encoding. tiktoken downloads these files on first use, so the first run needs internet access; after that they are cached. On a machine without internet, run once while online or point `TIKTOKEN_CACHE_DIR` at a pre-filled cache.

## Usage

Run all commands from the repository root.

On Windows, set `PYTHONUTF8=1` first. The scripts print emoji and Turkish characters; when their output is piped or redirected, Python falls back to the legacy code page and stops with `UnicodeEncodeError`.

```powershell
$env:PYTHONUTF8 = "1"        # cmd: set PYTHONUTF8=1
```

### basic

```bash
python basic/populate_database.py path/to/document.pdf --reset
python basic/query_data.py "Sorunuz"
```

The index is written to `chroma/<file name>/`. The query script lists the indexed documents and asks which one to use. Ingestion prints every chunk, which is useful for inspecting the splits.

### advanced

```bash
python advanced/populate_database.py path/to/document.pdf --reset
python advanced/query_rag.py "Sorunuz"
streamlit run advanced/app.py
```

The index is written to `advanced/chroma/<document slug>/`, separate from the basic version's indexes.

### Header and footer filtering

`UNWANTED` in each `populate_database.py` lists lines to strip from every page (repeated headers, footers, "this page intentionally left blank" notes). Add the strings that recur in your own documents.

## Example

One run (11 November 2026) on two public Yıldız Teknik Üniversitesi (YTÜ) documents, each indexed with `--reset` in both versions:

- Calendar: [YTÜ 2026-2027 undergraduate academic calendar (first-registration calendar)](https://ogi.yildiz.edu.tr/sites/ogi.yildiz.edu.tr/files/2026-2027-egitim-ogretim-yili-lisans-akademik-takvimi-ilk-kayit-takvimi.pdf), 1 page.
- Regulation: [YTÜ Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği](https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=15812&mevzuatTur=UniversiteYonetmeligi&mevzuatTertip=5), the consolidated text from mevzuat.gov.tr, 6 pages.

The answers below are copied verbatim from that run and were checked against the PDFs. Wrong answers are kept and marked.

| Question | basic | advanced |
|---|---|---|
| Calendar: Güz yarıyılı dersleri hangi tarihte başlıyor? | Güz yarıyılı dersleri 1 28 Eylül 2026 Pazartesi'nde başlıyor. | Güz yarıyılı derslerinin başlangıcı 1 Pazartesi'nden 28 Eylül 2026'ta oluyor. |
| Calendar: Güz dönemi katkı payı ödemeleri hangi tarihler arasında? | Güz dönemi katkı payı ödemeleri 14 Eylül-09 Ekim 2026 arasında yapılacaktır. | Cevap: Katkı payı ödeme sadece Güz döneminde yapılacaktır ve 14 Eylül-09 Ekim 2026 aralığındadır.<br>**Note:** the dates are right, but "sadece Güz döneminde" is not in the calendar. |
| Regulation: Lisans programları için azami öğrenim süresi kaç yıl? | Lisans programları için azami öğrenim süresi yedi yıldır. | Azami öğretim süresi lisans programlarında yedi yıldır. |
| Regulation: BA harf notunun katsayısı nedir? | Başarı notu katsayısı BA harfi için 3.00'dur.<br>**Wrong:** Madde 27 gives 3.50. | Üzgünüm, sorduğunuz sorunun cevabı belgede yok.<br>**Wrong:** the answer is in Madde 27. |
| Regulation: Yaz okulu ücreti ne kadar? | Üzgünüm, sorduğunuz sorunun cevabı belgede yok. | Üzgünüm, sorduğunuz sorunun cevabı belgede yok. |

- Both answers in the first row give the right date, 28 Eylül 2026; the stray "1" is the table's day-count cell.
- The last row is the expected result: the regulation mentions the summer school but states no fee.
- The advanced version extracted 18 table chunks from the calendar and 2 from the regulation. Both of the regulation's chunks are the amendment list on its last page. The letter-grade table in Madde 27 has no ruled lines in this PDF, so it was not extracted as a table; it is indexed as plain text with one cell per line. For the BA question the advanced version retrieved no context.

Models: LaBSE embeddings, the `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranker with sigmoid scores (advanced only), and Mistral 7B Instruct v0.3 (`mistral:v0.3`) through Ollama.

## Known limitations

The code is kept as it was built; these issues are documented rather than patched.

- **Self-check has no effect (`app.py`).** When the self-check finds missing items, the context is rebuilt, but `prompt.replace(context, context)` leaves the prompt unchanged. The three final answers are then generated from the original prompt, so the expanded context never reaches the answer.
- **Three-answer merge is redundant (`app.py`).** At temperature 0 the three answers are practically identical, so the merge step adds little.
- **App is slow (`app.py`).** It makes 6–7 LLM calls per question, which is slow on CPU. The cross-encoder is also loaded at module level, so Streamlit reloads it on every interaction.
- **Tables are indexed twice.** Camelot (lattice) and pdfplumber's default settings both detect ruled tables, so the same table is stored twice. Table text also remains in the page text.
- **Page numbers are inconsistent.** Text chunks use 0-based page numbers (from PyPDFLoader), while table chunks use 1-based ones.
- **Coverage loop can over-widen.** Question words are not filtered for stop words, so the loop may keep widening until it reaches the 180-candidate or 2,000-token limit.
- **DOCX is supported only by the basic version.**

## history/

- `populate_database_headings.py`: an experiment from between the two versions. It splits pages at headings, detected by font colour (PyMuPDF) or by numbered and all-caps lines (regex), and falls back to size-based chunks. It imports `langchain-chroma`, which is not in `requirements.txt`.
- `query_data_faiss.py`: a FAISS-based query experiment. The matching FAISS ingestion script was not kept, so it does not run as is.

## Credits

- The basic version's structure (separate populate/query/embedding scripts, `file:page:index` chunk IDs, Mistral through Ollama) follows pixegami's [rag-tutorial-v2](https://github.com/pixegami/rag-tutorial-v2).
- The idea of handling tables separately was inspired by Alejandro AO's [Multimodal RAG: Chat with PDFs (Images & Tables)](https://www.youtube.com/watch?v=uLrReyH5cu0). The implementation here differs: tables are extracted offline with Camelot and pdfplumber instead of being summarised by a cloud multimodal LLM.
- Models: [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), [Mistral 7B Instruct v0.3](https://ollama.com/library/mistral), and an MS MARCO cross-encoder.
