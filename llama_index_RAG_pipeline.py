from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer
from llama_index.core import ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Setting LLM
Settings.llm = Ollama("llama3", request_timeout=600)

# RAG PIPELINE STEPS
# 1 Load & Ingestion
documents = SimpleDirectoryReader("data").load_data()

# 2 Index & Embedding
print("embedding")
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("indexing")
index = VectorStoreIndex.from_documents(documents=documents)

# 3 Storing
# Not storing for while
# 4 Querying
query_engine = index.as_query_engine()
print("querying")
question = input("Your question: ")
response = query_engine.query(question)

print(response)