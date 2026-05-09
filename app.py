import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import os

from src.document_loader import load_pdf_documents
from src.text_processor import split_documents
from src.vector_store import create_vectorstore
from src.llm_service import answer_question
from src.feedback_store import save_interaction

load_dotenv()

st.set_page_config(page_title="AI Requirement & Document Assistant", layout="wide")
st.title("AI Requirement & Document Assistant")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

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

question = st.text_input("Ask a question from your uploaded documents")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process documents first.")
    else:
        with st.spinner("Generating answer..."):
            result = answer_question(st.session_state.vectorstore, question)

        st.session_state.last_result = result
        st.session_state.last_question = question

        save_interaction(
            question=question,
            answer=result["answer"],
            evidence=result["evidence"],
            sources=result["sources"],
            feedback="pending"
        )

if st.session_state.last_result:
    result = st.session_state.last_result

    st.subheader("Answer")
    st.write(result["answer"])

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