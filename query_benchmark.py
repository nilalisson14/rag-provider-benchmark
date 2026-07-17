import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

perguntas = [
    "Qual o prazo para a Agência analisar o processo de registro?",
    "O que acontece se a documentação estiver incompleta?",
    "Qual a validade do registro concedido?",
]

def montar_contexto(pergunta):
    docs = retriever.invoke(pergunta)
    return "\n\n".join(d.page_content for d in docs)

def perguntar(llm, pergunta, contexto, tentativas=3):
    prompt = f"Responda com base apenas no contexto abaixo.\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    for i in range(tentativas):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if i < tentativas - 1:
                print(f"  (tentativa {i+1} falhou, aguardando 10s...)")
                time.sleep(10)
            else:
                return f"[ERRO: {e}]"

llm_vertex = ChatVertexAI(
    model="gemini-2.5-flash",
    project="symbolic-heaven-502723-b4",
    location="us-central1"
)

llm_api = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

for pergunta in perguntas:
    contexto = montar_contexto(pergunta)
    print(f"\n=== Pergunta: {pergunta} ===")
    print(f"[Vertex AI] {perguntar(llm_vertex, pergunta, contexto)}")
    print(f"[Gemini API] {perguntar(llm_api, pergunta, contexto)}")