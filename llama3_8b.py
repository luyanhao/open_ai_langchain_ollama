# https://blog.csdn.net/sinat_29950703/article/details/136194337?app_version=6.3.1&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22136194337%22%2C%22source%22%3A%22cshao888%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama

prompt_template = "请给制作 {product} 的公司起个名字,只回答公司名即可，请注意：使用中文回答"

ollama_llm = Ollama(model="llama3:8b")
llm_chain = LLMChain(
    llm = ollama_llm,
    prompt = PromptTemplate.from_template(prompt_template)
)
print(llm_chain("袜子"))
# print(llm_chain.run("袜子"))
# print(llm_chain.predict("袜子"))