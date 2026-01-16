from rag.retriever import retrieve

query = "ethical concerns of AI in healthcare"

docs = retrieve(query)

print(f"Retrieved chunks: {len(docs)}\n")

for i, d in enumerate(docs, 1):
    print(f"--- CHUNK {i} ---")
    print("SOURCE:", d.metadata.get("source"))
    print("PAGE:", d.metadata.get("page", 0) + 1)
    print("TEXT:", d.page_content[:300])
    print()
