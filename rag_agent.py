import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (.env)
load_dotenv()

def run_rag_pipeline(query: str):
    print(f"\nUser Question: {query}")
    print("Searching database and generating response...\n")

    # Connect to embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Connect to vector database
    vector_db = Chroma(
        persist_directory="./chromadb_store",
        embedding_function=embeddings
    )

    # Retrieve matching chunks
    docs = vector_db.similarity_search(query, k=2)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Define prompt instructions
    prompt_template = """
    You are a professional assistant. 
    Answer the user question using only the context provided below.
    If you do not know the answer based on the context, say that you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Initialize Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.2
    )

    # Generate answer
    formatted_prompt = prompt.format(context=context_text, question=query)
    response = llm.invoke(formatted_prompt)

    print("Agent Final Response:")
    print("-" * 50)
    if isinstance(response.content, list):
        answer = "".join([block.get("text", "") for block in response.content if isinstance(block, dict)])
    else:
        answer = response.content

    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    test_question = " based on the Resume, What is the phone number of Faez and what is his cuurent job?"
    run_rag_pipeline(test_question)