import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import DATA_PATH, VECTORSTORE_PATH

load_dotenv()

def clean_text(text: str) -> str:
    """
    Basic text cleaning for RAG:
    - remove null characters
    - normalize whitespace
    - normalize newlines
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def load_documents():
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_PATH.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_PATH))


def main():
    docs = load_documents()

    chunks = split_documents(docs)
    print(f"Created chunks: {len(chunks)}")

    create_vectorstore(chunks)
    print("Vectorstore created successfully")


if __name__ == "__main__":
    main()
