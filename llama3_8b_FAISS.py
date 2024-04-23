# https://blog.csdn.net/WMX843230304WMX/article/details/137191360?app_version=6.3.1&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22137191360%22%2C%22source%22%3A%22cshao888%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# 从url导入知识作为聊天背景上下文
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
#加载
docs = loader.load()

# 文本分词器
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# ollama嵌入层
embeddings = OllamaEmbeddings()
embeddings.model = "llama3:8b"
# 文档向量化
vector = FAISS.from_documents(documents, embeddings)


# 创建ollama 模型
llm = Ollama(model="llama3:8b")
output_parser = StrOutputParser()

# 创建提示词模版
prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        请使用中文回答"""
    )
# 生成chain ：   prompt | llm 
document_chain = create_stuff_documents_chain(llm, prompt)

# 向量数据库检索器
retriever = vector.as_retriever()
#向量数据库检索chain :  vector | prompt | llm  
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 调用上面的 (向量数据库检索chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# 打印结果
print(response["answer"])
