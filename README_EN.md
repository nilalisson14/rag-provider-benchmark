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

## Results

### Qualitative comparison (answers)

| Question | Vertex AI | Gemini API |
|---|---|---|
| Review deadline | Correct (90 days) | Correct (90 days) |
| Incomplete documentation | Correct (30 days) | Correct (identical) |
| Registration validity | Correct (5 years) | Correct (5 years) |

For the same model (`gemini-2.5-flash`), answer quality doesn't change between Vertex AI and the Gemini API — the real difference lies in the access/billing layer, not in generation itself.

### RAGAS evaluation

| Metric | Vertex AI (Gemini 2.5 Flash) | Gemini API (Gemini 2.5 Flash) |
|---|---|---|
| Faithfulness | 1.00 | Not completed (see finding below) |
| Answer relevancy | 0.85 | Not completed |
| Context precision | 1.00 | Not completed |

### Technical findings

**1. Gemini API billing change (March 2026)**
Since March 2026, Gemini API usage is no longer covered by the Google Cloud Free Trial's $300 credit — it requires its own prepaid balance in AI Studio. This changes how "free tier" should be understood when planning a zero-budget RAG project: the Google Cloud/Vertex AI path (corporate billing) proved more stable and immediately usable than the standalone Gemini API path.

**2. RAGAS + Gemini API + Windows incompatibility**
RAGAS evaluation ran end-to-end without issues using Vertex AI as both generator and judge. When attempting to evaluate the Gemini API side — even isolated in its own script, without Vertex AI in the same process — the process consistently crashed (`RuntimeError: Event loop is closed`, `TimeoutError`) before completing all evaluation jobs, even after: updating/pinning RAGAS versions, fixing a broken import in the library itself (a known bug, see [issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745)), forcing the `WindowsSelectorEventLoopPolicy` event loop policy, and reducing concurrency (`max_workers=1`). The pattern suggests a low-level incompatibility between the async `google-genai`/gRPC client and RAGAS's event loop on Windows — not a project configuration error. Documented here as a finding, without blocking the benchmark's completion.

## Current status

- [x] End-to-end RAG pipeline working (corpus → chunking → embeddings → ChromaDB → generation)
- [x] Vertex AI side validated with 100% accuracy on test questions
- [x] Gemini API side validated qualitatively (correct answers, manual comparison)
- [x] Complete RAGAS evaluation for the Vertex AI side
- [x] Documented finding: RAGAS + Gemini API + Windows incompatibility
- [ ] RAGAS evaluation for the Gemini API side (blocked by the incompatibility above; possible future fix: run on Linux/WSL)
- [ ] Test corpus is still synthetic and small (1 document, 6 chunks) — needs expansion to be more representative

## Next steps

1. Test the Gemini API side's RAGAS evaluation on WSL or Linux, where the default event loop shouldn't have the same conflict.
2. Expand the synthetic corpus to more documents, closer to a regulated-domain scenario (ANVISA/SNCR).
3. Write and publish the LinkedIn post covering both findings (billing and technical incompatibility).
