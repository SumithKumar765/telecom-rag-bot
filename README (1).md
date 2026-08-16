# 📡 3GPP Technical Specifications RAG Assistant

## 📝 Overview
This project is a high-accuracy Retrieval-Augmented Generation (RAG) system designed to query complex 3GPP telecom standards (specifically TS 38.300 for 5G NR Architecture). Built with a focus on enterprise-grade reliability and zero hallucination, the assistant leverages a dual-LLM fallback architecture. It strictly grounds all answers in the provided engineering documents, ensuring technical precision for telecom professionals and researchers.

## ✨ Key Features
- **Zero-Hallucination Guardrails:** Strict system prompts and deterministic generation (`temperature=0.0`) prevent the LLM from inventing outside information.
- **Dual-LLM Resiliency:** Uses Groq's high-speed inference as the primary engine, with an automatic failover to Google Gemini to handle rate limits and ensure maximum uptime.
- **Optimized Semantic Search:** Employs the `BAAI/bge-small-en-v1.5` model (384 dimensions) with normalized embeddings for highly accurate cosine similarity matching of technical telecom acronyms.
- **Auditable Citations:** An interactive Streamlit UI allows users to expand and inspect the exact source chunks and document pages used to generate the response.
- **Cloud-Ready:** Includes a smart SQLite patch for seamless deployment to Streamlit Community Cloud or Linux-based servers.

## 🏗️ System Architecture

| Component | Technology | Specification |
| :--- | :--- | :--- |
| **Knowledge Base** | PDF | 3GPP TS 38.300 (5G NR Architecture) |
| **Framework** | LangChain | `create_retrieval_chain`, `.with_fallbacks()` |
| **Embeddings** | HuggingFace | `BAAI/bge-small-en-v1.5` (384-dim, CPU optimized) |
| **Vector Store** | ChromaDB | Local persistence, `k=4` retrieval |
| **Primary LLM** | Groq | `llama-3.3-70b-versatile` |
| **Fallback LLM** | Google Gemini | `gemini-2.5-flash` |
| **Frontend** | Streamlit | Conversational UI with state management |

---

## 🚀 Setup & Installation Guide

### 1. Prerequisites
- Python 3.9+
- A free [Groq API Key](https://console.groq.com/keys)
- A free [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/telecom-rag-bot.git
cd telecom-rag-bot
```

### 3. Create a Virtual Environment
```bash
python -m venv venv

# On Windows
.\venv\Scripts\Activate.ps1

# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory of the project and add your API keys:
```env
GROQ_API_KEY="your_groq_api_key_here"
GOOGLE_API_KEY="your_google_api_key_here"
```

### 6. Build the Vector Database
Place your 3GPP document (`TS_38_300.pdf`) into a `data/` directory, then run the ingestion script to chunk the document and build the ChromaDB vector store.
```bash
python ingest.py
```
> **Note:** This will download the embedding model weights (~130MB) on the first run.

### 7. Run the Application
Launch the Streamlit user interface:
```bash
streamlit run app.py
```
The application will automatically open in your browser at `http://localhost:8501`.

---

## 🧪 Testing the Guardrails
To verify the system's accuracy and anti-hallucination constraints, try the following queries in the UI:

1. **Domain Technicality:** *"What are the main functions of the gNB in NG-RAN architecture?"* — Should return a precise, cited answer.
2. **Out-of-Scope Blocking:** *"What is the recipe for a chocolate cake?"* — Should trigger the fallback refusal response.
3. **Failover Simulation:** Temporarily change your Groq API key to an invalid string in your `.env` file and ask a question. The app will seamlessly catch the error, route to Google Gemini, and still return an accurate answer.

---

## 📄 License
Add your license here (e.g., MIT, Apache 2.0).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.
