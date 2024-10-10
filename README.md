# Play with LLMs using llamaindex

This shows RAG pipeline.

The [llama_index_single_RAG_pipeline.py](llama_index_single_RAG_pipeline.py) shows single idea how to run RAG pipeline  

## Requirements
- [Ollama](https://ollama.com/)
- [Docker](https://www.docker.com/)

## Seeting up

### Python Setup
```bash
python3 -m venv .venv
source .venv
pip install -r requirements
```

### Downloading ollama model
```bash
ollama pull llama3
```

## Running
```bash
python3 llama_index_single_RAG_pipeline.py
```