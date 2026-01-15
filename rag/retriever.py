from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import VECTORSTORE_PATH, TOP_K

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
    """
    Retrieve top-K relevant document chunks for a query.

    Optional:
    - source_filter: restrict results to a specific document
    """
    if source_filter:
        docs = _vectorstore.similarity_search(
            query=query,
            k=TOP_K,
            filter={"source": source_filter}
        )
    else:
        docs = _vectorstore.similarity_search(
            query=query,
            k=TOP_K
        )

    return docs
