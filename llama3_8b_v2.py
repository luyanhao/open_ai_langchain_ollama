from langchain_community.llms import Ollama
llm = Ollama(model="llama3:8b")
print(llm.invoke("天为什么是蓝色的？请使用中文回答"))