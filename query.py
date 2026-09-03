import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Configure stdout encoding to utf-8 on Windows to avoid UnicodeEncodeError with emojis
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

def search_knowledge_base(user_query: str):
    """
    Performs a Similarity Search in ChromaDB using Gemini Embeddings.
    """
    print(f"\n🔍 Searching for: '{user_query}'...\n")
    
    # Initialize the same Gemini Embeddings model used during ingestion
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Connect to the existing ChromaDB store directory
    db_path = "./chromadb_store"
    if not os.path.exists(db_path):
        print("⚠️ Vector store not found! Please run 'store_embeddings.py' first.")
        return
        
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Perform Similarity Search (Retrieve top 2 matching chunks)
    results = vector_db.similarity_search(user_query, k=2)
    
    # Display the retrieved relevant chunks
    print(f"✅ Found {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print("-" * 30)

if __name__ == "__main__":
    # Test query
    sample_question = "What ERP systems has Faez worked with?"
    search_knowledge_base(sample_question)  