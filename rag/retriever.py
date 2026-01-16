from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import (
    VECTORSTORE_PATH,
    TOP_K,
    DISTANCE_THRESHOLD,
)

load_dotenv()

_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

_vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    _embeddings,
    allow_dangerous_deserialization=True
)

def retrieve(
    query: str,
    source_filter: str | None = None
):

    docs_with_scores = _vectorstore.similarity_search_with_score(
        query=query,
        k=TOP_K,
        filter={"source": source_filter} if source_filter else None
    )

    relevant_docs = [
        doc
        for doc, score in docs_with_scores
        if score < DISTANCE_THRESHOLD
    ]

    return relevant_docs
