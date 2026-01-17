import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag.qa import answer_question

st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    layout="centered"
)

st.title("🩺 Healthcare RAG Chatbot")
st.caption(
    "Answers are generated strictly from the uploaded healthcare documents. "
    "If the answer is not found, the system will respond with I don't know."
)

st.divider()

question = st.text_input(
    "Ask a healthcare-related question:",
    placeholder="e.g. What risks are associated with digital health?"
)

if question:
    with st.spinner("Searching documents and generating answer..."):
        answer, citations = answer_question(question)

    st.markdown("### Answer")
    st.markdown(
        f"""
        <div style="
            background-color:#111827;
            padding:16px;
            border-radius:10px;
            border-left:4px solid #2563eb;
        ">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    if citations:
        st.markdown("### Sources")
        for c in citations:
            st.markdown(f"- {c}")
