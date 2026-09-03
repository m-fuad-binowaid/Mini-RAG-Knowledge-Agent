import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def start_interactive_chat():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    vector_db = Chroma(
        persist_directory="./chromadb_store",
        embedding_function=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional knowledge assistant. Answer the user strictly using the provided context. If the answer is not contained in the context, state that you do not know.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chat_history = []
    print("\n--- Conversational Agent Ready (Type 'exit' to quit) ---\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "خروج"]:
            print("Session ended.")
            break

        retrieved_docs = vector_db.similarity_search(user_input, k=2)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        chain = prompt | llm
        response = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": user_input
        })

        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer_text])

        print(f"\nAgent:\n{answer_text}\n")
        print("-" * 50)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer_text))

if __name__ == "__main__":
    start_interactive_chat()
    