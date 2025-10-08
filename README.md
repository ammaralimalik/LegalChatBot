# LegalChatBot (Pakistan Law RAG)

A local Retrieval-Augmented Generation (RAG) chatbot focused on Pakistani law. It ingests legal PDFs (e.g., Constitution, Penal Code, Contract Act), builds a persistent Chroma vector store using Legal-BERT embeddings, and answers queries via a local LLM API.


## Features
- Legal PDF ingestion with heading-aware chunking
- Persistent Chroma DB collection (`Base_Books`)
- Embeddings using `nlpaueb/legal-bert-base-uncased`
- Streamlit chat UI
- Local LLM backend (tested with `deepseek-r1-distill-qwen-7b` via OpenAI-compatible API)


## Project Structure
- `books/`: Put your legal PDFs here
- `doc_ingestion.py`: Extracts text, chunks, and ingests embeddings into Chroma
- `embedding_model.py`: Legal-BERT embedding function for Chroma
- `database.py`: Queries Chroma for top-k relevant chunks
- `llm_connection.py`: Calls local LLM API (OpenAI-compatible) and cleans chain-of-thought
- `streamlit_files/app.py`: Streamlit chat frontend
- `chroma.sqlite3` and UUID dirs: Chroma persistence artifacts


## Requirements
Python 3.13 (project venv included under `ChatBot/`, but you can use your own)

Install dependencies:
```bash
pip install -r requirements.txt
```
`requirements.txt` (summary): chromadb, PyPDF2, langchain_text_splitters, transformers, torch

Note: `torch` install may need platform-specific wheels. See PyTorch site if default install fails.


## Quick Start
1) Prepare PDFs
- Place `.pdf` files in `books/`

2) Start/verify local LLM server
- Expects an OpenAI-compatible endpoint at `http://127.0.0.1:1234/v1/chat/completions`
- Model name used: `deepseek-r1-distill-qwen-7b`
- Adjust in `llm_connection.py` if needed

3) Ingest documents into Chroma
```bash
python doc_ingestion.py
```
This will:
- Parse each PDF with PyPDF2
- Split text by legal headings, then chunk with RecursiveCharacterTextSplitter
- Generate embeddings with Legal-BERT
- Add to Chroma collection `Base_Books`

4) Run the chat UI
```bash
streamlit run streamlit_files/app.py
```
Open the provided local URL and ask a legal question.


## Configuration
- Chroma persistence path:
  - `doc_ingestion.py` uses `CHROMA_PATH = '/LegalChatbot/'`
  - `database.py` uses an absolute path: `/Users/ammarmalik/Desktop/ResumeProjects/LegalChatBot/`
  - For portability, set both to the same path. Recommended: project root absolute path. Example:
    - In `doc_ingestion.py`: `CHROMA_PATH = '/Users/ammarmalik/Desktop/ResumeProjects/LegalChatBot/'`
    - In `database.py`: keep the same absolute path

- Collection name: `Base_Books` (change in both `doc_ingestion.py` and `database.py`)

- Embedding model: `nlpaueb/legal-bert-base-uncased` (configured in `embedding_model.py`)

- LLM endpoint & model: configured in `llm_connection.py`
  - Endpoint: `http://127.0.0.1:1234/v1/chat/completions`
  - Model: `deepseek-r1-distill-qwen-7b`
  - System prompt enforces Pakistan legal context; adjust as needed


## How It Works (RAG Flow)
1) Ingestion (`doc_ingestion.py`)
   - Extract text from PDFs
   - Split by legal headings (regex on Chapter/Section/Article/etc.), then chunk with overlap
   - Persist embeddings to Chroma (collection `Base_Books`)

2) Retrieval (`database.py`)
   - For a user prompt, query Chroma for top 3 relevant chunks

3) Generation (`streamlit_files/app.py` + `llm_connection.py`)
   - Build a prompt with retrieved context
   - Call local LLM API and stream reply in Streamlit
   - Remove any `<think>...</think>` content


## Tips & Gotchas
- Ensure the Chroma path is consistent across `doc_ingestion.py` and `database.py` or you’ll see empty results.
- The ingestion loop sleeps 45s per file (rate considerations). Adjust `time.sleep(45)` if needed.
- Large PDFs may take time for embedding; watch console logs like `Ingested <file>:<id>`.
- Streamlit warning about `torch._classes`: the app clears module path to avoid import errors.


## Troubleshooting
- No results/empty context:
  - Verify PDFs exist in `books/` and ingestion ran without errors
  - Confirm both scripts point to the same Chroma path and collection name

- Torch or model load errors:
  - Install correct `torch` wheel for your OS/arch
  - Check internet access for downloading `nlpaueb/legal-bert-base-uncased`

- LLM not responding:
  - Ensure your local API server is running and reachable
  - Confirm endpoint and model name in `llm_connection.py`

- Streamlit not launching:
  - Run from project root and use the exact command shown above


## Roadmap
- Add chat history persistence
- Add citations and source highlighting per chunk
- Add evaluation scripts for retrieval quality
- Containerize app and models for easier setup


## License
This is a personal portfolio project; review and adapt licensing before production use.
