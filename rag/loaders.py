from langchain_community.document_loaders import PyPDFLoader
from rag.cleaning import clean_text

def load_documents(data_path):
    """
    Load and clean PDF documents.
    """
    documents = []

    for pdf_path in data_path.glob("*.pdf"):
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
