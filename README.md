# Healthcare RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** built with **LangChain, OpenAI, FAISS, and Streamlit**.  
The system answers healthcare-related questions **strictly based on a curated set of documents**, provides
**transparent citations**, and safely handles *"I don't know"* cases.

This project is developed as part of **Giga Academy Cohort IV – Project #4: RAG Chatbot**.

---

## Features

- Modular document ingestion pipeline (load → clean → chunk → embed → index)
- Semantic retrieval using **FAISS vector search (Top-K)**
- Answer generation grounded **only in retrieved document context**
- Automatic source citations (document name + page reference)
- Safe refusal handling when information is not found
- Short-term conversation memory (context-aware follow-up questions)
- Optional metadata-based filtering by document
- Hallucination-safe responses
- Simple and clean **Streamlit UI**

---

## Knowledge Base

Curated corpus of ~40 healthcare-related documents

**Document types include:**

- Clinical decision support reviews
- Primary health care and health systems research
- AI implementation and governance in healthcare

Documents are publicly available, academic, and non-patient-specific

All answers are generated exclusively from this document set

**Dataset location:**

```
data/raw_docs/healthcare/
```

## Project Structure

```
rag-chatbot/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI for the RAG chatbot
│
├── data/
│   └── raw_docs/
│       └── healthcare/         # Healthcare PDF documents (knowledge base)
│
├── rag/
│   ├── vectorstore/
│   │   ├── index.faiss         # FAISS vector index
│   │   └── index.pkl           # FAISS index metadata
│   │
│   ├── __init__.py
│   ├── loaders.py              # PDF loading logic
│   ├── cleaning.py             # Text cleaning & normalization
│   ├── chunking.py             # Document chunking logic
│   ├── vectorstore.py          # FAISS index creation
│   ├── retriever.py            # Vector similarity retrieval
│   ├── qa.py                   # Question answering pipeline
│   ├── prompts.py              # RAG prompt templates
│   ├── config.py               # Global configuration
│   └── ingest.py               # Ingestion pipeline orchestration
│
├── .env                        # Environment variables (API keys)
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```

---

## Setup Instructions

### 1. Clone the Repository & Create Virtual Environment

```bash
git clone https://github.com/RinesaGerxhaliu/rag-chatbot.git
cd rag-chatbot
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-openai`
- `pypdf`
- `python-docx`
- `tiktoken`
- `faiss-cpu`
- `numpy`
- `pandas`
- `python-dotenv`

### 3. Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Dataset Setup

**Note:**  
The healthcare documents used for this project are already included in the repository.
No additional downloads are required after cloning.

Dataset location:
```
data/raw_docs/healthcare/
```

### 5. Run the Ingestion Pipeline

This step:
- Loads and cleans documents
- Splits them into chunks
- Creates and saves a FAISS vector index

```bash
python -m rag.ingest
```

Expected output:
```
Loaded pages: XXX
Chunks created: XXX
Vector store created successfully
```

### 6. Run the Chatbot (Streamlit UI)

```bash
streamlit run app/streamlit_app.py
```

Then open your browser at: `http://localhost:8501`

---

## How It Works (High Level)

1. The user submits a healthcare-related question
2. The retriever finds the Top-K most relevant chunks
3. The LLM generates an answer strictly from retrieved context
4. Sources are shown with document name + page number
5. If the answer is not found → "I don't know based on the provided documents."

---

## Example Questions

- What risks are associated with digital health systems?
- How is AI used in clinical decision support?
- What challenges exist in healthcare data governance?
- Which pages mention this topic?

---

## Safety & Guardrails

- The prompt explicitly forbids using external knowledge
- Instructions inside documents are ignored
- The system never guesses or hallucinates answers

---

## Author

**Rinesa Gerxhaliu**  
AI Engineering – Giga Academy Cohort IV
