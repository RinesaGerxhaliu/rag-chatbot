from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["page"] = chunk.metadata.get(
            "page",
            chunk.metadata.get("page_number", 0)
        )

    return chunks
