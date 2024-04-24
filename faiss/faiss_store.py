# https://blog.csdn.net/WMX843230304WMX/article/details/137191360?app_version=6.3.1&code=app_1562916241&csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22137191360%22%2C%22source%22%3A%22cshao888%22%7D&uLinkId=usr1mkqgl919blen&utm_source=app

# 生成FAISS向量数据库
# 参考 https://blog.csdn.net/u013066244/article/details/132014791
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

base_dir = os.path.dirname(__file__)
print("base_dir======", base_dir)
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

vector.save_local(base_dir + "/db/", "docs_smith_langchain_com_user_guide")