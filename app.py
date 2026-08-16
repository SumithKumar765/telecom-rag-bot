import os
import sys
import streamlit as st

# --- SMART SQLITE PATCH FOR STREAMLIT CLOUD ---
# This safely ignores the patch on your Windows PC but applies it on the Linux Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 
# ----------------------------------------------

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (.env)
load_dotenv()

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="3GPP Telecom RAG Assistant",
    page_icon="📡",
    layout="wide"
)

# --- Sidebar Metadata & Architecture Overview ---
with st.sidebar:
    st.header("📡 System Architecture")
    st.markdown("""
    **Core Pipeline Specs:**
    - **Knowledge Base:** 3GPP TS 38.300 (5G NR Architecture)
    - **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dim short dense embeddings)
    - **Vector Store:** ChromaDB (Cosine Similarity, $k=4$)
    - **Primary LLM:** Groq (`llama-3.3-70b-versatile`)
    - **Failover / Backup LLM:** Google (`gemini-2.5-flash`)
    - **Temperature:** `0.0` (Deterministic / Factual)
    """)
    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("📡 3GPP Technical Specifications RAG Assistant")
st.caption("Zero-Hallucination QA Engine grounded strictly in Telecom 3GPP Release Standards.")

# --- 1. RAG Pipeline Initialization (Cached for Low Latency) ---
@st.cache_resource(show_spinner="Initializing Embeddings & Vector Database...")
def load_rag_pipeline():
    # 1. Short embedding model with normalization for optimal cosine distance
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Connect to local ChromaDB
    if not os.path.exists("chroma_db"):
        st.error("Vector database 'chroma_db' not found. Please run 'python ingest.py' first.")
        st.stop()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Primary LLM: Groq (Llama 3.3 70B - fast inference)
    # max_retries=0 ensures instant failure switch to backup if rate limits are hit
    primary_llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_retries=0
    )

    # 4. Fallback LLM: Google Gemini (High stability, massive token context)
    backup_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0
    )

    # 5. Resilient LLM routing with automatic failover
    resilient_llm = primary_llm.with_fallbacks([backup_llm])

    # 6. Anti-Hallucination System Prompt
    system_prompt = (
        "You are an expert telecom standards engineer assisting with 3GPP technical queries.\n"
        "Strictly answer the user's question using ONLY the provided 3GPP context below.\n"
        "Rules:\n"
        "1. If the provided context does not contain sufficient facts to answer the question, state: "
        "'I do not have enough information in the provided 3GPP documents to answer this.'\n"
        "2. Do NOT guess, extrapolate, or rely on pre-trained outside knowledge.\n"
        "3. Maintain technical precision (cite acronyms, interfaces, and clauses where present).\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 7. Construct retrieval-augmented QA chain
    qa_chain = create_stuff_documents_chain(resilient_llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# Initialize the chain
chain = load_rag_pipeline()

# --- 2. Chat Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved 3GPP Source Chunks"):
                st.markdown(message["sources"])

# --- 3. Query Handling & Generation ---
if user_query := st.chat_input("Ask about 5G NR architecture, NG-RAN, or protocol layers..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate grounded response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant 3GPP specifications..."):
            try:
                response = chain.invoke({"input": user_query})
                answer = response["answer"]
                source_docs = response.get("context", [])

                # Format source chunks with metadata for inspection
                source_text = ""
                for idx, doc in enumerate(source_docs, start=1):
                    page_num = doc.metadata.get("page", "Unknown")
                    source_file = os.path.basename(doc.metadata.get("source", "3GPP Doc"))
                    snippet = doc.page_content.strip().replace("\n", " ")
                    source_text += f"**[Chunk {idx}] - File:** `{source_file}` | **Page:** `{page_num}`\n\n> {snippet}\n\n---\n"

                # Render response and source expander
                st.markdown(answer)
                if source_text:
                    with st.expander("🔍 View Retrieved 3GPP Source Chunks"):
                        st.markdown(source_text)

                # Persist to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_text
                })

            except Exception as e:
                error_msg = f"**Pipeline Execution Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": ""
                })