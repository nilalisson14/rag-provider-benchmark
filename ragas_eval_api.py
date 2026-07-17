import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset
import time
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
retriever = db.as_retriever(search_kwargs={"k": 3})
perguntas = ["Qual o prazo para a Agencia analisar o processo de registro?", "O que acontece se a documentacao estiver incompleta?", "Qual a validade do registro concedido?"]
respostas_esperadas = ["O prazo e de ate 90 dias corridos, contados a partir do protocolo completo da documentacao.", "A Agencia notifica o requerente, que tem 30 dias para complementar, sob pena de arquivamento.", "O registro tem validade de 5 anos, podendo ser renovado."]
def montar_contexto(pergunta):
    docs = retriever.invoke(pergunta)
    return [d.page_content for d in docs]
def perguntar(llm, pergunta, contexto_lista, tentativas=3):
    contexto = "\n\n".join(contexto_lista)
    prompt = f"Responda com base apenas no contexto abaixo.\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    for i in range(tentativas):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if i < tentativas - 1:
                time.sleep(10)
            else:
                return f"[ERRO: {e}]"
llm_api = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
contextos, respostas = [], []
for p in perguntas:
    ctx = montar_contexto(p)
    contextos.append(ctx)
    respostas.append(perguntar(llm_api, p, ctx))
ds_api = Dataset.from_dict({"question": perguntas, "answer": respostas, "contexts": contextos, "ground_truth": respostas_esperadas})
judge_llm = LangchainLLMWrapper(llm_api)
judge_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
metricas = [faithfulness, answer_relevancy, context_precision]
config = RunConfig(max_workers=1, timeout=120)
print("Avaliando Gemini API (isolado)...")
resultado_api = evaluate(ds_api, metrics=metricas, llm=judge_llm, embeddings=judge_embeddings, run_config=config)
print("Gemini API:", resultado_api)