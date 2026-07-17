from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project="symbolic-heaven-502723-b4",
    location="us-central1"
)

response = llm.invoke("Diga apenas 'conectado com sucesso' se você recebeu esta mensagem.")
print(response.content)