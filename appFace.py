import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from ingest import process_and_split_documents

load_dotenv()

DATA_DIR = "data"
DB_DIR = "./chromadb_store"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

st.set_page_config(
    page_title="Knowledge Agent",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0d0f14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #12151d;
        border-right: 1px solid #1f2430;
    }
    
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 1rem;
    }
    
    .chat-row.user {
        justify-content: flex-end;
    }
    
    .chat-row.assistant {
        justify-content: flex-start;
    }
    
    .chat-bubble {
        max-width: 85%;
        padding: 12px 18px;
        border-radius: 12px;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .chat-bubble.user {
        background-color: #1c2333;
        border: 1px solid #2d3748;
        color: #f7fafc;
        border-bottom-right-radius: 4px;
    }
    
    .chat-bubble.assistant {
        background-color: #161922;
        border: 1px solid #232834;
        color: #e2e8f0;
        border-bottom-left-radius: 4px;
    }

    .role-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        color: #94a3b8;
    }
    
    .role-title.user {
        text-align: right;
        color: #818cf8;
    }

    .source-line {
        font-size: 0.82rem;
        color: #cbd5e1;
        padding: 4px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

def rebuild_vector_database():
    chunks = process_and_split_documents()
    embeddings = get_embeddings_model()

    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    
    try:
        vector_db.delete_collection()
    except Exception:
        pass

    if chunks:
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
    return vector_db

def get_vector_store():
    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=get_embeddings_model()
        )
    return None

@st.cache_resource
def get_llm_and_prompt():
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly intelligent knowledge assistant. 
Answer user questions using the provided context.

Key Guidelines:
1. Ground your facts strictly on the provided context.
2. Use logical reasoning and common sense deductions (for example, in naming conventions such as Arabic names, a person's second name represents their father's name).
3. If information cannot be found or deduced with high certainty from the text, state clearly that it is not mentioned.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    return llm, prompt

llm, prompt = get_llm_and_prompt()

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("### Document Controls")
    existing_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    filter_options = ["All Documents"] + existing_files
    selected_scope = st.selectbox("Search Target:", filter_options)

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Documents:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.button("Index Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Indexing..."):
                for file in uploaded_files:
                    target_path = os.path.join(DATA_DIR, file.name)
                    with open(target_path, "wb") as f:
                        f.write(file.getbuffer())
                rebuild_vector_database()
                st.rerun()

    st.divider()

    if existing_files:
        st.markdown("**Current Files**")
        for file_name in existing_files:
            col_name, col_btn = st.columns([0.8, 0.2])
            with col_name:
                st.caption(file_name)
            with col_btn:
                if st.button("×", key=f"del_{file_name}"):
                    file_path = os.path.join(DATA_DIR, file_name)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    rebuild_vector_database()
                    st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==================== Main Interface ====================
st.markdown("### ✨ Knowledge Agent")
st.caption(f"Active scope: `{selected_scope}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = msg["role"]
    display_title = "You" if role == "user" else "Agent"
    
    st.markdown(
        f"""
        <div class="chat-row {role}">
            <div class="chat-bubble {role}">
                <div class="role-title {role}">{display_title}</div>
                <div>{msg["content"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if role == "assistant" and "sources" in msg and msg["sources"]:
        with st.expander("Retrieved Sources"):
            for src in msg["sources"]:
                st.markdown(f"<div class='source-line'>📌 <b>Source:</b> <code>{src['source']}</code> &nbsp;|&nbsp; <b>Match Score:</b> {src['score']}</div>", unsafe_allow_html=True)

if user_input := st.chat_input("Ask anything from your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(
        f"""
        <div class="chat-row user">
            <div class="chat-bubble user">
                <div class="role-title user">You</div>
                <div>{user_input}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    vector_db = get_vector_store()

    if not vector_db or not existing_files:
        warning_msg = "No documents available to query. Please index documents via the sidebar."
        st.markdown(
            f"""
            <div class="chat-row assistant">
                <div class="chat-bubble assistant">
                    <div class="role-title">Agent</div>
                    <div>{warning_msg}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.messages.append({"role": "assistant", "content": warning_msg})
    else:
        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))

        search_filter = None
        if selected_scope != "All Documents":
            search_filter = {"source": os.path.join(DATA_DIR, selected_scope)}

        docs_with_scores = vector_db.similarity_search_with_score(
            user_input, 
            k=3, 
            filter=search_filter
        )

        sources_data = []
        context_chunks = []

        for doc, score in docs_with_scores:
            context_chunks.append(doc.page_content)
            source_name = doc.metadata.get("source", "Unknown Document")
            relevance_percentage = f"{max(0.0, 1.0 - score) * 100:.1f}%"
            
            sources_data.append({
                "source": os.path.basename(source_name),
                "score": relevance_percentage
            })

        context_text = "\n\n".join(context_chunks)

        chain = prompt | llm
        response = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "question": user_input
        })

        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer_text])

        st.markdown(
            f"""
            <div class="chat-row assistant">
                <div class="chat-bubble assistant">
                    <div class="role-title">Agent</div>
                    <div>{answer_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("Retrieved Sources"):
            for src in sources_data:
                st.markdown(f"<div class='source-line'>📌 <b>Source:</b> <code>{src['source']}</code> &nbsp;|&nbsp; <b>Match Score:</b> {src['score']}</div>", unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "sources": sources_data
        })
        st.rerun()