import os
import re
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
    - Normalize whitespace
    - Normalize newlines
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def load_documents():
    """
    Load and clean PDF documents from the dataset directory.

    Each PDF is loaded page-by-page. Text is cleaned and metadata
    (source filename) is attached to each page.
    """
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(DATA_PATH / file)
            pages = loader.load()

            for page in pages:
                page.page_content = clean_text(page.page_content)
                page.metadata["source"] = file

            documents.extend(pages)

    return documents


def split_documents(documents):
    """
    Split documents into overlapping chunks for semantic retrieval.

    Chunking improves retrieval accuracy and ensures that
    context fits within model limits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    """
    Create and persist a FAISS vector store from document chunks.
    Embeddings are generated using OpenAI embeddings and stored
    locally for efficient similarity search.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_PATH.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))


def main():
    """
    Steps:
    1. Load and clean documents
    2. Split documents into chunks
    3. Create FAISS vector store
    """
    documents = load_documents()
    print(f"Loaded documents: {len(documents)}")

    chunks = split_documents(documents)
    print(f"Created chunks: {len(chunks)}")

    create_vectorstore(chunks)
    print("Vectorstore created successfully")


if __name__ == "__main__":
    main()
