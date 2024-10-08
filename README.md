# Play with LLMs using llamaindex

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

## Runningh
```bash
python3 llama_index_RAG_pipeline.py
```