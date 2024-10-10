from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode

from llama_index.llms.llama_cpp import LlamaCPP
import psycopg2


from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List

from llama_index.core.query_engine import RetrieverQueryEngine


DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_TABLE = "data"
DB_DATABASE="vector_db"

# SENTENCE TRANSFORMER
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

#LLAMA CCP

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True
)

# INITIALIZE POSTGRES

conn = psycopg2.connect(
    dbname=DB_NAME,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER
)

conn.autocommit = True

# with conn.cursor() as c:
#     c.execute(f"DROP DATABASE IF EXISTS {DB_DATABASE}")
#     c.execute(f"CREATE DATABASE {DB_DATABASE}")

vector_store = PGVectorStore.from_params(
    database=DB_DATABASE,
    host=DB_HOST,
    password=DB_PASSWORD,
    port=DB_PORT,
    user=DB_USER,
    table_name=DB_TABLE,
    embed_dim=384,
)

# STEP 1

loader = PyMuPDFReader()
documents = loader.load(file_path="./data_pdf/llama2.pdf")
# documents = loader.load(file_path="./data_pdf/amazon.pdf")

# STEP 2

text_parser = SentenceSplitter(
    chunk_size=1024
)

text_chunks = []
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chuncks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chuncks)
    doc_idxs.extend([doc_idx]*len(cur_text_chuncks))

# STEP 3

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

# STEP 4

for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode=MetadataMode.ALL)
    )
    node.embedding = node_embedding

# STEP 5

vector_store.add(nodes)

model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)

# BUILD RETRIEVAL PIPELINE FROM SCRATCH

query_str = "What are LLMs?"

# 1 GENERATE A QUERY EMBEDDING

query_embedding = embed_model.get_query_embedding(query_str)

# 2 QUERY THE VECTOR DATABASE

query_mode = "default"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)

query_result = vector_store.query(vector_store_query)
print(f"Query result = {query_result.nodes[0].get_content()}")

# 3 PARSE RESULT INTO A SET OF NODES

nodes_with_scores = []
for index, node in enumerate(query_result.nodes):
    score: Optional[float] = None
    if query_result.similarities is not None:
        score = query_result.similarities[index]
    nodes_with_scores.append(NodeWithScore(node=node, score=score))

# 4 PUT INTO A RETRIEVER

# class VectorDBRetriever(BaseRetriever):
#     """Retriever over a postgres vector store."""
#
#     def __init__(
#         self,
#         vector_store: PGVectorStore,
#         embed_model: Any,
#         query_mode: str = "default",
#         similarity_top_k: int = 2,
#     ) -> None:
#         """Init params."""
#         self._vector_store = vector_store
#         self._embed_model = embed_model
#         self._query_mode = query_mode
#         self._similarity_top_k = similarity_top_k
#         super().__init__()
#
#     def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
#         """Retrieve."""
#         query_embedding = embed_model.get_query_embedding(
#             query_bundle.query_str
#         )
#         vector_store_query = VectorStoreQuery(
#             query_embedding=query_embedding,
#             similarity_top_k=self._similarity_top_k,
#             mode=self._query_mode,
#         )
#         query_result = vector_store.query(vector_store_query)
#
#         nodes_with_scores = []
#         for index, node in enumerate(query_result.nodes):
#             score: Optional[float] = None
#             if query_result.similarities is not None:
#                 score = query_result.similarities[index]
#             nodes_with_scores.append(NodeWithScore(node=node, score=score))
#
#         return nodes_with_scores