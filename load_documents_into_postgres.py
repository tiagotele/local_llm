import psycopg2
import os

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer

DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = 'postgres'
DB_HOST = 'localhost'
DB_PORT = 5432

# Set locall llama
# embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Settings.embed_model = embedding_model

def connect_to_db():
    connection = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return  connection

def insert_item(connection, description, embedding):
    
    with connection.cursor() as cursor:
        embedding_str = str(embedding)
        cursor.execute(
            "INSERT INTO items_test (description, embedding) VALUES (%s, %s)",
            (description, embedding_str)
        )
        connection.commit()

if __name__ == "__main__":
    print("main")
    files_in_folder = [f for f in os.listdir("data") ]
    print(files_in_folder)
    try: 
        for f in files_in_folder:
            print(f"file = {f}")
            with open(f"data/{f}", "r") as file:
                file_content = file.readlines()
                embedded_str = str(embedding_model.encode(file_content[0]).tolist())
                # embedded_str = embedded_str.replace(" ", "")
                print("\n\n\n\n")
                # print(embedded_str.tolist())
                # print(f)
                insert_item(connect_to_db(), f, embedded_str)
    except Exception as e:
        print(e)