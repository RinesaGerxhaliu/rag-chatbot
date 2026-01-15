import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import (
    DATA_PATH,
    VECTORSTORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

load_dotenv()


def clean_text(text: str) -> str:
    """
    Operations:
    - Remove null characters
    - Normalize line breaks
    - Normalize spaces (preserve paragraphs)
    - Remove References / Bibliography sections (if at end)
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lower_text = text.lower()
    for keyword in ["references", "bibliography"]:
        idx = lower_text.rfind(keyword)
        if idx != -1 and idx > len(text) * 0.7:
            text = text[:idx]

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_documents():
    """
    Each page:
    - Cleaned using clean_text
    - Annotated with source metadata
    """
    documents = []

    for pdf_path in DATA_PATH.glob("*.pdf"):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            cleaned = clean_text(page.page_content)
            if not cleaned:
                continue

            page.page_content = cleaned
            page.metadata["source"] = pdf_path.name
            documents.append(page)

    return documents

def split_documents(documents):
    """
    Split documents into overlapping chunks for semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """
    Create and persist a FAISS vector store from document chunks.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))

def main():
    """
    Ingestion pipeline:
    1. Load and clean documents
    2. Split into chunks
    3. Create FAISS vector store
    """
    documents = load_documents()
    print(f"Loaded pages: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    create_vectorstore(chunks)
    print("Vector store created successfully")

if __name__ == "__main__":
    main()
