from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag.retriever import retrieve
from rag.prompts import QA_PROMPT
from rag.config import MODEL_TEMPERATURE

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=MODEL_TEMPERATURE
)

def is_prompt_injection(question: str) -> bool:
    patterns = [
        "ignore previous",
        "ignore instructions",
        "act as",
        "system prompt",
        "developer message",
        "change your role",
    ]
    q = question.lower()
    return any(p in q for p in patterns)

def is_page_question(question: str) -> bool:
    keywords = [
        "which page",
        "on which page",
        "what page",
        "which pages",
    ]
    q = question.lower()
    return any(k in q for k in keywords)

def answer_question(question: str, source_filter: str | None = None):
    if is_prompt_injection(question):
        return "I can only answer questions based on the provided healthcare documents.", []

    docs = retrieve(question, source_filter)

    if not docs:
        return "I don't know based on the provided documents.", []

    if is_page_question(question):
        pages_by_source = {}

        for d in docs:
            src = d.metadata.get("source")
            page = d.metadata.get("page", 0) + 1
            pages_by_source.setdefault(src, set()).add(page)

        answer = []
        citations = []

        for src, pages in pages_by_source.items():
            plist = ", ".join(str(p) for p in sorted(pages))
            answer.append(f"In **{src}**, this topic appears on pages {plist}.")
            for p in pages:
                citations.append(f"{src} (page {p})")

        return "\n".join(answer), citations

    context = "\n\n".join(d.page_content for d in docs)
    prompt = QA_PROMPT.format(context=context, question=question)
    answer = llm.invoke(prompt).content.strip()

    normalized = answer.lower()

    if "don't know" in normalized or "do not know" in normalized:
        return "I don't know based on the provided documents.", []

    citations = list({
        f"{d.metadata.get('source')} (page {d.metadata.get('page', 0) + 1})"
        for d in docs
    })

    return answer, citations
