from dotenv import load_dotenv
from rag.config import DATA_PATH, VECTORSTORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

from rag.loaders import load_documents
from rag.chunking import split_documents
from rag.vectorstore import create_vectorstore

load_dotenv()

def main():
    documents = load_documents(DATA_PATH)
    print(f"Loaded pages: {len(documents)}")

    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created chunks: {len(chunks)}")

    create_vectorstore(chunks, VECTORSTORE_PATH)
    print("Vector store created successfully")

if __name__ == "__main__":
    main()
