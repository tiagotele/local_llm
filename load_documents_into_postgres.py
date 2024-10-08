import logging
import os

import psycopg2
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(),
    ],
)

# Create a logger object
logger = logging.getLogger(__name__)

DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432
FOLDER_WITH_DATA = "data"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def connect_to_db(
    db_name: str,
    db_user: str,
    db_password: str,
    db_host: str = "localhost",
    db_port: int = 5432,
):
    connection = psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port
    )
    return connection


def insert_item(connection, file_name, embedding):

    with connection.cursor() as cursor:
        embedding_str = str(embedding)
        cursor.execute(
            "INSERT INTO items_test (description, embedding) VALUES (%s, %s)",
            (file_name, embedding_str),
        )
        connection.commit()
        logger.info(f"File {file_name} inserted.")


def load_files_from_folder(folder_name: str = "data"):

    files_in_folder = [file for file in os.listdir(folder_name)]

    for file in files_in_folder:

        with open(f"{folder_name}/{file}", "r") as file_reader:
            file_content = file_reader.readlines()
            embedded = embedding_model.encode(file_content[0]).tolist()
            insert_item(
                connect_to_db(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT),
                file,
                str(embedded),
            )


if __name__ == "__main__":
    try:
        load_files_from_folder(FOLDER_WITH_DATA)
    except Exception as e:
        logger.error(f"Error inserting file: {e}")
        raise e
