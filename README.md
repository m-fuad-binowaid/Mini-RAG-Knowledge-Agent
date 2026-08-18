# Mini-RAG Knowledge Agent

A minimal RAG system that answers questions based on local documents.

> 🚧 Work in progress

## Stack

- Google Gemini (embeddings + generation)
- ChromaDB (vector store)
- LangChain

## Setup

```bash
pip install -r requirements.txt
cp _env .env  # add your GOOGLE_API_KEY
```

## Usage

```bash
python store_embeddings.py  # index your documents
python rag_agent.py         # ask a question
```