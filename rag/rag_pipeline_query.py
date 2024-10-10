from typing import Optional

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


def query_vector_db(query_str:str , embed_model: HuggingFaceEmbedding, vector_store: PGVectorStore)-> Optional[str|None]:
    query_embedding = embed_model.get_query_embedding(query_str)

    query_mode = "default"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )

    query_result = vector_store.query(vector_store_query)
    return query_result.nodes[0].get_content()




if __name__ == '__main__':
    DB_NAME = "postgres"
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_TABLE = "data"
    DB_DATABASE = "postgres"

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    vector_store = PGVectorStore.from_params(
        database=DB_DATABASE,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=DB_TABLE,
        embed_dim=384,
    )

    query_str = "What was the profit in Amazon in 2023?"
    answer = query_vector_db(query_str, embed_model, vector_store)
    print(f"Answer from Vector DB: {answer}")
    