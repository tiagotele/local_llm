import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode


def ingest_file(path:str, file_name:str, embed_model:HuggingFaceEmbedding, vector_store: PGVectorStore) -> None:
    loader = PyMuPDFReader()
    documents = loader.load(file_path=f"./{path}/{file_name}")

    text_parser = SentenceSplitter(
        chunk_size=1024
    )

    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chuncks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chuncks)
        doc_idxs.extend([doc_idx]*len(cur_text_chuncks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode=MetadataMode.ALL)
        )
        node.embedding = node_embedding

    vector_store.add(nodes)

def load_files_from_folder(folder_name: str, embed_model:HuggingFaceEmbedding, vector_store: PGVectorStore):
    files = [file for file in os.listdir(folder_name)]
    for file in files:
        ingest_file(folder_name, file, embed_model, vector_store)

if __name__ == "__main__":
    db_name = "postgres"
    db_user = "postgres"
    db_password = "postgres"
    db_host = "localhost"
    db_port = "5432"
    db_table = "data"
    db_database = "postgres"
    folder_with_data = "data_pdf"

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    vect_store = PGVectorStore.from_params(
        database=db_database,
        host=db_host,
        password=db_password,
        port=db_port,
        user=db_user,
        table_name=db_table,
        embed_dim=384,
    )
    load_files_from_folder(folder_with_data, embed_model, vect_store)
    print(f"Files from {folder_with_data} loaded successfully.")