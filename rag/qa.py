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

def answer_question(question: str):
    docs = retrieve(question)

    if not docs:
        return "I don't know based on the provided documents.", []

    context = "\n\n".join(d.page_content for d in docs)
    prompt = QA_PROMPT.format(context=context, question=question)

    answer = llm.invoke(prompt).content.strip()

    if "i don't know" in answer.lower():
        return "I don't know based on the provided documents.", []

    citations = list({
        f"{d.metadata.get('source')} (page {d.metadata.get('page', 0) + 1})"
        for d in docs
    })

    return answer, citations
