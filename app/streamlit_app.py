import sys
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag.qa import answer_question
from rag.config import DATA_PATH

st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }

      .chat-bubble {
        padding: 14px 18px;
        border-radius: 14px;
        margin-bottom: 12px;
        line-height: 1.6;
        max-width: 85%;
        word-wrap: break-word;
      }

      .user-msg {
        background-color: #1f2937;
        border-right: 4px solid #3b82f6;
        margin-left: auto;
        font-size: 0.9rem;
        opacity: 0.9;
      }

      .assistant-msg {
        background-color: #0f172a;
        border-left: 4px solid #10b981;
        margin-right: auto;
        font-size: 0.95rem;
      }

      .role-label {
        font-size: 0.75rem;
        opacity: 0.65;
        margin: 2px 6px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Healthcare RAG Chatbot")
st.caption(
    "Answers are generated strictly from the uploaded healthcare documents. "
    "If the answer is not found, the system responds with *I don't know*."
)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

with st.sidebar:
    st.markdown("### Document Filter")

    documents = ["All documents"] + [p.name for p in DATA_PATH.glob("*.pdf")]
    selected_doc = st.selectbox(
        "Restrict answers to:",
        documents
    )

    st.caption("Optional. Leave as *All documents* for global search.")

for m in st.session_state.messages:
    role_class = "user-msg" if m["role"] == "user" else "assistant-msg"
    role_label = "Question" if m["role"] == "user" else "Answer"

    st.markdown(
        f'<div class="role-label">{role_label}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="chat-bubble {role_class}">
            {m["content"]}
        </div>
        """,
        unsafe_allow_html=True
    )

    if m["role"] == "assistant" and m.get("citations"):
        with st.expander("Sources"):
            for c in m["citations"]:
                st.markdown(f"- {c}")

question = st.chat_input("Ask a healthcare-related question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.spinner("Searching documents and generating answer..."):
        source_filter = None if selected_doc == "All documents" else selected_doc

        if st.session_state.last_question:
            combined_question = (
                f"Previous question: {st.session_state.last_question}\n"
                f"Current question: {question}"
            )
        else:
            combined_question = question

        answer, citations = answer_question(combined_question, source_filter)

        st.session_state.last_question = question
        st.session_state.last_answer = answer

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })

    st.rerun()
