import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import os
import html
from src.document_loader import load_pdf_documents_grouped
from src.vector_store import create_vectorstores_for_documents
from src.llm_service import compare_documents

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
    
if "vectorstores_by_doc" not in st.session_state:
    st.session_state.vectorstores_by_doc = {}

mode = st.sidebar.selectbox(
    "Choose mode",
    ["Q&A", "Summary", "Requirements Extraction", "Document Comparison"]
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Loading and indexing documents..."):
            grouped_docs = load_pdf_documents_grouped(uploaded_files)

            grouped_chunks = {}
            all_chunks = []

            for filename, docs in grouped_docs.items():
                chunks = split_documents(docs)
                grouped_chunks[filename] = chunks
                all_chunks.extend(chunks)

            st.session_state.vectorstore = create_vectorstore(all_chunks)
            st.session_state.vectorstores_by_doc = create_vectorstores_for_documents(grouped_chunks)

        st.success("Documents processed successfully!")

if mode == "Q&A":
    user_input = st.text_input("Ask a question from your uploaded documents")
elif mode == "Summary":
    user_input = st.text_area("Write what you want summarized, or leave it blank")
else:
    user_input = st.text_area("Write what requirements you want extracted, or leave it blank")

if mode == "Document Comparison":
    available_docs = list(st.session_state.vectorstores_by_doc.keys())

    if len(available_docs) >= 2:
        col1, col2 = st.columns(2)

        with col1:
            doc_a = st.selectbox("Select Document A", available_docs)

        with col2:
            default_index = 1 if len(available_docs) > 1 else 0
            doc_b = st.selectbox("Select Document B", available_docs, index=default_index)

        compare_query = st.text_input("What do you want to compare?")

        if compare_query:
            with st.spinner("Comparing documents..."):
                result = compare_documents(
                    st.session_state.vectorstores_by_doc[doc_a],
                    st.session_state.vectorstores_by_doc[doc_b],
                    compare_query
                )

            st.subheader("Comparison Result")
            st.markdown(result["answer"])

            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Sources from Document A")
                for i, item in enumerate(result["sources_a"], start=1):
                    with st.expander(f"A{i}: {item['source']} | Page {item['page']}"):
                        st.markdown(f"**Similarity distance:** {item['score']}")
                        st.write(item["snippet"])

            with col_b:
                st.subheader("Sources from Document B")
                for i, item in enumerate(result["sources_b"], start=1):
                    with st.expander(f"B{i}: {item['source']} | Page {item['page']}"):
                        st.markdown(f"**Similarity distance:** {item['score']}")
                        st.write(item["snippet"])
    else:
        st.info("Please upload at least two PDF documents for comparison.")

else:
    if mode == "Q&A":
        user_input = st.text_input("Ask a question from your uploaded documents")
    elif mode == "Summary":
        user_input = st.text_area("Write what you want summarized")
    else:
        user_input = st.text_area("Write what requirements you want extracted")

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

        st.subheader("Output")
        st.markdown(result["answer"])

        st.subheader("Evidence Level")
        if result["evidence"] == "High":
            st.success("High evidence")
        elif result["evidence"] == "Medium":
            st.warning("Medium evidence")
        else:
            st.error("Low evidence")

        st.subheader("Sources")
        if result["sources"]:
            for i, item in enumerate(result["sources"], start=1):
                with st.expander(f"Source {i}: {item['source']} | Page {item['page']}"):
                    st.markdown(f"**Similarity distance:** {item['score']}")
                    st.write(item["snippet"])
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