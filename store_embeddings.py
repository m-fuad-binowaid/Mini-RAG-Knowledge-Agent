import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from ingest import process_and_split_documents

load_dotenv()

DB_DIRECTORY = "./chromadb_store"

def create_and_store_all_embeddings():
    """Generates embeddings for all chunks and persists them in ChromaDB."""
    print("Step 1: Reading and chunking all documents...")
    chunks = process_and_split_documents()

    if not chunks:
        print("Embedding process aborted: No chunks available.")
        return

    print("\nStep 2: Generating vector embeddings with Gemini...")
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
)

    print(f"Step 3: Storing vectors in directory: {DB_DIRECTORY}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY
    )

    print("\nSuccess: All documents embedded and stored in ChromaDB.")

if __name__ == "__main__":
    create_and_store_all_embeddings()