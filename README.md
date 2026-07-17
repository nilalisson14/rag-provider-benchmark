# RAG Provider Benchmark

Comparação prática entre **Vertex AI (Gemini)** e **Gemini API** rodando o mesmo pipeline RAG, avaliadas sobre o mesmo corpus e as mesmas perguntas.

## Contexto

Projeto de portfólio criado para explorar, na prática, a diferença entre consumir modelos Gemini via **Google Cloud / Vertex AI** (autenticação por projeto, billing corporativo) e via **Gemini API** (autenticação por chave, billing prepago separado desde março de 2026). O objetivo é entender, com dados reais, o que muda operacionalmente entre as duas formas de acesso ao mesmo modelo — não apenas ler a documentação, mas rodar e medir.

## Arquitetura

```
Corpus de documentos (texto sintético)
        │
        ▼
Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings (sentence-transformers/all-MiniLM-L6-v2, local)
        │
        ▼
Vector store (ChromaDB, persistido em ./chroma_db)
        │
        ▼
Pipeline RAG (retriever top-k=3)
        │
        ├──────────────┐
        ▼              ▼
  Vertex AI       Gemini API
  Gemini 2.5      Gemini 2.5
  Flash           Flash
  (projeto GCP)   (API key)
        │              │
        └──────┬───────┘
               ▼
     Relatório comparativo
     (respostas, custo, disponibilidade)
```

**Por que embeddings locais**: ao manter o mesmo modelo de embeddings para os dois lados, a única variável que muda entre as duas execuções é o modelo de geração (o LLM), não a recuperação de contexto. Isso torna a comparação mais justa.

## Stack

- **Orquestração**: LangChain (`langchain`, `langchain-text-splitters`, `langchain-community`)
- **Vector store**: ChromaDB
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (execução local, sem custo)
- **LLM — lado Google Cloud**: `ChatVertexAI` (`langchain-google-vertexai`), modelo `gemini-2.5-flash`
- **LLM — lado Gemini API**: `ChatGoogleGenerativeAI` (`langchain-google-genai`), modelo `gemini-2.5-flash`
- **Ambiente**: Python 3.14, venv, Windows (CMD)

## Passo a passo executado

### 1. Configuração do Google Cloud
1. Criação de conta no Google Cloud e vinculação de conta de faturamento ao projeto (`symbolic-heaven-502723-b4`).
2. Ativação da API de modelos (rebrandizada em abril/2026 de "Vertex AI API" para **"Agent Platform API"**, parte do Gemini Enterprise Agent Platform — mesma infraestrutura por trás, nome novo na interface).
3. Instalação do Google Cloud CLI (`gcloud`).
4. Autenticação local via Application Default Credentials:
   ```
   gcloud init
   gcloud auth application-default login
   ```

### 2. Ambiente Python
```
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-community chromadb sentence-transformers langchain-google-genai langchain-google-vertexai
```

### 3. Teste de conectividade (Vertex AI)
Script isolado (`teste_vertex.py`) validando autenticação e resposta do modelo antes de montar o pipeline completo.

### 4. Ingestão (`ingest.py`)
Carrega o corpus, faz chunking (500 caracteres, overlap 50), gera embeddings locais e persiste no ChromaDB.

### 5. Benchmark comparativo (`query_benchmark.py`)
Roda as mesmas 3 perguntas contra o mesmo índice vetorial, alternando entre `ChatVertexAI` e `ChatGoogleGenerativeAI`, com retry simples (3 tentativas, 10s de espera) para lidar com instabilidade momentânea das APIs.

## Resultados

### Comparação qualitativa (respostas)

| Pergunta | Vertex AI | Gemini API |
|---|---|---|
| Prazo de análise | Correto (90 dias) | Correto (90 dias) |
| Documentação incompleta | Correto (30 dias) | Correto (idêntico) |
| Validade do registro | Correto (5 anos) | Correto (5 anos) |

Para o mesmo modelo (`gemini-2.5-flash`), a qualidade da resposta não muda entre Vertex AI e Gemini API — a diferença real está na camada de acesso e billing, não na geração em si.

### Avaliação RAGAS

| Métrica | Vertex AI (Gemini 2.5 Flash) | Gemini API (Gemini 2.5 Flash) |
|---|---|---|
| Faithfulness | 1.00 | Não concluído (ver achado abaixo) |
| Answer relevancy | 0.85 | Não concluído |
| Context precision | 1.00 | Não concluído |

### Achados técnicos

**1. Mudança de billing da Gemini API (março/2026)**
Desde março de 2026, o uso da Gemini API deixou de ser coberto pelo crédito de $300 do Google Cloud Free Trial — exige saldo prepago próprio no AI Studio. Isso muda a forma como "free tier" deve ser entendido ao planejar um projeto de RAG com orçamento zero: o caminho via Google Cloud/Vertex AI (billing corporativo) se mostrou mais estável e imediato do que o caminho via Gemini API standalone.

**2. Incompatibilidade RAGAS + Gemini API + Windows**
A avaliação RAGAS rodou de ponta a ponta sem problemas usando Vertex AI como gerador e como juiz. Ao tentar avaliar o lado Gemini API — mesmo isolando-o em um script próprio, sem o Vertex AI no mesmo processo — o processo consistentemente travou (`RuntimeError: Event loop is closed`, `TimeoutError`) antes de completar todos os jobs de avaliação, mesmo após: atualizar/fixar versões do RAGAS, corrigir um import quebrado na própria biblioteca (bug conhecido, ver [issue #2745](https://github.com/vibrantlabsai/ragas/issues/2745)), forçar a política de event loop `WindowsSelectorEventLoopPolicy`, e reduzir a concorrência (`max_workers=1`). O padrão sugere uma incompatibilidade de baixo nível entre o cliente assíncrono `google-genai`/gRPC e o event loop do RAGAS no Windows — não um erro de configuração do projeto. Ficou documentado aqui como achado, sem bloquear a conclusão do benchmark.

## Status atual

- [x] Pipeline RAG funcional ponta a ponta (corpus → chunking → embeddings → ChromaDB → geração)
- [x] Lado Vertex AI validado com 100% de acerto nas perguntas de teste
- [x] Lado Gemini API validado qualitativamente (respostas corretas, comparação manual)
- [x] Avaliação RAGAS completa para o lado Vertex AI
- [x] Achado documentado: incompatibilidade RAGAS + Gemini API + Windows
- [ ] Avaliação RAGAS para o lado Gemini API (bloqueada pela incompatibilidade acima; possível solução futura: rodar em ambiente Linux/WSL)
- [ ] Corpus de teste ainda é sintético e pequeno (1 documento, 6 chunks) — expandir para um corpus mais representativo

## Próximos passos

1. Testar a avaliação RAGAS do lado Gemini API em WSL ou Linux, onde o event loop padrão não deve ter o mesmo conflito.
2. Expandir o corpus sintético para mais documentos, aproximando do cenário de domínio regulado (ANVISA/SNCR).
3. Escrever e publicar o post no LinkedIn com os dois achados (billing e incompatibilidade técnica).
