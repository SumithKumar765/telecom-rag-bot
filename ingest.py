import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    # 1. Load the 3GPP Document
    print("Loading 3GPP Document...")
    pdf_path = "data/TS_38_300.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Check your data folder.")
        return

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # 2. Split text into manageable chunks
    print("Chunking document...")
    # Using 1000 chunk size and 200 overlap to prevent splitting technical acronyms
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} overlapping chunks.")

    # 3. Load a State-of-the-Art short embedding model
    print("Loading BAAI/bge-small-en-v1.5 embedding model...")
    # model_kwargs ensures it uses CPU efficiently
    # encode_kwargs normalizes embeddings to boost cosine similarity accuracy
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True} 
    )

    # 4. Generate embeddings and store them locally in ChromaDB
    print("Building Vector Database (this may take 1-3 minutes depending on your CPU)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Success! Vector database built and saved to the 'chroma_db' folder.")

if __name__ == "__main__":
    build_vector_db()