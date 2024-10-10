# Local RAG pipeline with llama_index and Postgres.

## Pipeline

### Ingestion
The file [rag_pipeline.load_into_database.py](rag_pipeline_load_into_database.py) loads PDFs files content, 
embed it with HuggingFace and finally stores it into Postgres using Vector plugin.

### Query
For query data stored in Postgres just run [rag_pipeline_query.py](rag_pipeline_query.py). It will parse the query 
string into embedding, and finally will query into database using VectorStoreQuery.