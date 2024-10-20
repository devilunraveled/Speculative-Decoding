from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model = "llama3.1:8b")

def getResponse(query):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", query),
        ])
    output = model.invoke({"input" : prompt})
    print(output.logits)
