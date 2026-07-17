# RAG Provider Benchmark

A practical comparison between **Vertex AI (Gemini)** and the **Gemini API**, running the same RAG pipeline, evaluated on the same corpus and the same questions.

## Context

Portfolio project built to explore, hands-on, the difference between consuming Gemini models through **Google Cloud / Vertex AI** (project-based authentication, corporate billing) versus the **Gemini API** (API-key authentication, separate prepaid billing since March 2026). The goal is to understand, with real data, what changes operationally between the two ways of accessing the same underlying model — not just reading the docs, but actually running and measuring it.

## Architecture

```
Document corpus (synthetic text)
        │
        ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings (sentence-transformers/all-MiniLM-L6-v2, local)
        │
        ▼
Vector store (ChromaDB, persisted in ./chroma_db)
        │
        ▼
RAG pipeline (retriever, top-k=3)
        │
        ├──────────────┐
        ▼              ▼
  Vertex AI       Gemini API
  Gemini 2.5      Gemini 2.5
  Flash           Flash
  (GCP project)   (API key)
        │              │
        └──────┬───────┘
               ▼
     Comparative report
     (answers, cost, availability)
```

**Why local embeddings**: keeping the same embedding model on both sides means the only variable that changes between the two runs is the generation model (the LLM), not context retrieval. This makes the comparison fairer.

## Stack

- **Orchestration**: LangChain (`langchain`, `langchain-text-splitters`, `langchain-community`)
- **Vector store**: ChromaDB
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no cost)
- **LLM — Google Cloud side**: `ChatVertexAI` (`langchain-google-vertexai`), model `gemini-2.5-flash`
- **LLM — Gemini API side**: `ChatGoogleGenerativeAI` (`langchain-google-genai`), model `gemini-2.5-flash`
- **Environment**: Python 3.14, venv, Windows (CMD)

## Step-by-step

### 1. Google Cloud setup
1. Created a Google Cloud account and linked a billing account to the project (`symbolic-heaven-502723-b4`).
2. Enabled the model API (rebranded in April 2026 from "Vertex AI API" to **"Agent Platform API"**, part of the Gemini Enterprise Agent Platform — same underlying infrastructure, new name in the console).
3. Installed the Google Cloud CLI (`gcloud`).
4. Authenticated locally via Application Default Credentials:
   ```
   gcloud init
   gcloud auth application-default login
   ```

### 2. Python environment
```
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community chromadb sentence-transformers langchain-google-genai langchain-google-vertexai
```

### 3. Connectivity test (Vertex AI)
Standalone script (`teste_vertex.py`) validating authentication and model response before building the full pipeline.

### 4. Ingestion (`ingest.py`)
Loads the corpus, chunks it (500 characters, 50 overlap), generates local embeddings, and persists them to ChromaDB.

### 5. Comparative benchmark (`query_benchmark.py`)
Runs the same 3 questions against the same vector index, alternating between `ChatVertexAI` and `ChatGoogleGenerativeAI`, with a simple retry (3 attempts, 10s wait) to handle momentary API instability.

## Results (first run)

| Provider | Result |
|---|---|
| Vertex AI (Gemini 2.5 Flash) | 3/3 questions answered correctly, citing the right articles from the corpus |
| Gemini API (Gemini 2.5 Flash) | Blocked by `429 RESOURCE_EXHAUSTED` — free tier quota exhausted, then "prepayment credits are depleted" |

### Key finding
Since March 2026, Gemini API usage is no longer covered by the Google Cloud Free Trial's $300 credit — it requires its own prepaid balance in AI Studio. This changes how "free tier" should be understood when planning a zero-budget RAG project: the Google Cloud/Vertex AI path (corporate billing) proved more stable and immediately usable than the standalone Gemini API path.

## Current status

- [x] End-to-end RAG pipeline working (corpus → chunking → embeddings → ChromaDB → generation)
- [x] Vertex AI side validated with 100% accuracy on test questions
- [ ] Gemini API side pending prepaid balance to complete the comparison
- [ ] RAGAS metrics (faithfulness, answer relevancy, context precision) not yet implemented
- [ ] Test corpus is still synthetic and small (1 document, 6 chunks) — needs expansion to be more representative

## Next steps

1. Load a small prepaid balance into the Gemini API, or compare models within Vertex AI itself (e.g. `gemini-2.5-flash` vs `gemini-2.5-flash-lite`) in the meantime.
2. Implement RAGAS evaluation over both sides' answers.
3. Generate a final comparative table and chart (cost, latency, answer quality).
4. Expand the synthetic corpus to more documents, closer to a regulated-domain scenario (ANVISA/SNCR).
