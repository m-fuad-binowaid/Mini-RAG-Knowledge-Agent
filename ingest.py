import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

def load_and_split_document(file_path: str):
    """
    Reads a text document and splits it into smaller chunks for processing.
    """
    print(f"📖 Reading document from: {file_path}...")
    
    # 1. Load the text document
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # 2. Configure the text splitter (Chunk size: 500 characters, Overlap: 50 characters)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # 3. Split the document into chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Successfully split the document into {len(chunks)} chunks!\n")
    
    # Display the first two chunks as a preview
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print("-" * 25)
        
    return chunks

if __name__ == "__main__":
    sample_file = os.path.join("data", "sample.txt")
    
    if os.path.exists(sample_file):
        load_and_split_document(sample_file)
    else:
        print(f"⚠️ File not found! Please create {sample_file}")
