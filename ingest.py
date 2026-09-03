import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DATA_DIR = "data"

def load_single_document(file_path: str):
    """Loads a single file depending on its extension."""
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        return loader.load()
    else:
        print(f"Skipping unsupported file: {file_path}")
        return []

def load_all_documents(folder_path: str = DATA_DIR):
    """Scans the directory and loads all supported documents."""
    all_documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}. Please place your files inside it.")
        return []

    files = os.listdir(folder_path)
    print(f"Found {len(files)} files in '{folder_path}'.")

    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        if os.path.isfile(full_path):
            print(f"Loading: {file_name}...")
            docs = load_single_document(full_path)
            all_documents.extend(docs)

    return all_documents

def process_and_split_documents():
    """Loads all documents and splits them into clean chunks."""
    documents = load_all_documents(DATA_DIR)
    
    if not documents:
        print("No documents found to process.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nSuccessfully generated {len(chunks)} chunks from all files.")
    return chunks

if __name__ == "__main__":
    process_and_split_documents()