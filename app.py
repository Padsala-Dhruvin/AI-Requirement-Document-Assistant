import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import os
import html

from src.document_loader import load_pdf_documents
from src.text_processor import split_documents
from src.vector_store import create_vectorstore
from src.llm_service import run_mode
from src.feedback_store import save_interaction

load_dotenv()

st.set_page_config(page_title="AI Requirement & Document Assistant", layout="wide")
st.title("AI Requirement & Document Assistant")

st.markdown(
    """
    <style>
        .answer-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-left: 5px solid #4f46e5;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            color: #111827;
            line-height: 1.7;
            font-size: 1.02rem;
        }
        .answer-card ol.answer-list {
            margin: 0;
            padding-left: 1.35rem;
        }
        .answer-card ol.answer-list li {
            margin-bottom: 0.55rem;
        }
        .source-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.78rem;
            font-weight: 700;
            color: #6b7280;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_answer_block(answer_text):
    if isinstance(answer_text, list):
        lines = []
        for item in answer_text:
            if isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or str(item)
                lines.append(str(text_value).strip())
            else:
                lines.append(str(item).strip())
    elif isinstance(answer_text, dict):
        lines = [str(answer_text.get("text") or answer_text.get("content") or answer_text).strip()]
    else:
        lines = [line.strip() for line in str(answer_text).splitlines() if line.strip()]

    if len(lines) > 1:
        items_html = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
        return f"<ol class='answer-list'>{items_html}</ol>"

    return f"<div>{html.escape(lines[0] if lines else str(answer_text))}</div>"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

mode = st.sidebar.selectbox(
    "Choose mode",
    ["Q&A", "Summary", "Requirements Extraction"]
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Loading and indexing documents..."):
            documents = load_pdf_documents(uploaded_files)
            chunks = split_documents(documents)
            st.session_state.vectorstore = create_vectorstore(chunks)
        st.success("Documents processed successfully!")

if mode == "Q&A":
    user_input = st.text_input("Ask a question from your uploaded documents")
elif mode == "Summary":
    user_input = st.text_area("Write what you want summarized, or leave it blank")
else:
    user_input = st.text_area("Write what requirements you want extracted, or leave it blank")

if user_input:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first.")
    else:
        with st.spinner("Running mode..."):
            result = run_mode(st.session_state.vectorstore, user_input, mode=mode)

        st.session_state.last_result = result
        st.session_state.last_question = user_input

        save_interaction(
            question=user_input,
            answer=result["answer"],
            evidence=result["evidence"],
            sources=result["sources"],
            feedback="pending"
        )

if st.session_state.last_result:
    result = st.session_state.last_result

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Evidence", result["evidence"])
    with metric_col2:
        st.metric("Sources", len(result["sources"]))
    with metric_col3:
        st.metric("Mode", mode)

    st.markdown("<div class='section-label'>Answer</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='answer-card'>{render_answer_block(result['answer'])}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-label' style='margin-top:1.2rem;'>Sources</div>", unsafe_allow_html=True)
    if result["sources"]:
        for i, item in enumerate(result["sources"], start=1):
            with st.container():
                st.markdown(
                    f"""
                    <div class='source-card'>
                        <strong>Source {i}:</strong> {item['source']}<br>
                        <strong>Page:</strong> {item['page']}<br>
                        <strong>Similarity distance:</strong> {item['score']}<br>
                        <div style='margin-top:0.6rem; color:#374151;'>{item['snippet']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No supporting sources found.")

    st.subheader("Feedback")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful"):
            save_interaction(
                question=st.session_state.last_question,
                answer=result["answer"],
                evidence=result["evidence"],
                sources=result["sources"],
                feedback="helpful"
            )
            st.success("Feedback saved: Helpful")

    with col2:
        if st.button("👎 Not Helpful"):
            save_interaction(
                question=st.session_state.last_question,
                answer=result["answer"],
                evidence=result["evidence"],
                sources=result["sources"],
                feedback="not_helpful"
            )
            st.warning("Feedback saved: Not Helpful")

st.divider()
st.subheader("Feedback Log Preview")

feedback_path = "data/feedback_log.csv"
if os.path.exists(feedback_path):
    df = pd.read_csv(feedback_path)

    st.write(f"Total logged interactions: {len(df)}")

    if "feedback" in df.columns:
        helpful_count = (df["feedback"] == "helpful").sum()
        not_helpful_count = (df["feedback"] == "not_helpful").sum()
        pending_count = (df["feedback"] == "pending").sum()

        st.markdown(f"- Helpful: {helpful_count}")
        st.markdown(f"- Not Helpful: {not_helpful_count}")
        st.markdown(f"- Pending: {pending_count}")

    st.dataframe(df.tail(10), use_container_width=True)
else:
    st.info("No feedback log created yet.")