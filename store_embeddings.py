import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure stdout encoding to utf-8 on Windows to avoid UnicodeEncodeError with emojis
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

def create_and_store_embeddings():
    """
    Reads local document chunks, generates numerical vector embeddings 
    using Google Gemini, and persists them into ChromaDB.
    """
    file_path = os.path.join("data", "sample.txt")
    
    print("📖 Step 1: Loading and splitting document...")
    loader = TextLoader(file_path, encoding='utf-8')    
    documents = loader.load()
    
    # Split text into chunks (500 chars with 50 char overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} document chunks.")

    print("\n🧠 Step 2: Generating Embeddings & Storing in ChromaDB...")
    
    # Initialize Google Gemini Embeddings Model with a supported model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Generate embeddings and store them locally in ./chromadb_store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chromadb_store"
    )
    
    print("🎉 All document chunks successfully embedded and saved to ChromaDB!")

if __name__ == "__main__":
    create_and_store_embeddings()