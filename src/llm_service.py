import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL
from src.prompt_templates import (
    get_qa_prompt,
    get_summary_prompt,
    get_requirements_prompt,
    get_compare_prompt
)
from src.vector_store import retrieve_documents_with_scores

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2
    )

def clean_snippet(text, max_chars=300):
    text = " ".join(text.split())
    return text[:max_chars] + "..." if len(text) > max_chars else text

def build_context(scored_docs):
    parts = []
    for idx, (doc, score) in enumerate(scored_docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        snippet = clean_snippet(doc.page_content, 500)

        parts.append(
            f"[{idx}] Source: {source} | Page: {page + 1 if isinstance(page, int) else page}\n{snippet}"
        )
    return "\n\n".join(parts)

def get_evidence_label(scored_docs):
    if not scored_docs:
        return "Low"

    best_score = scored_docs[0][1]

    if best_score < 0.8:
        return "High"
    elif best_score < 1.5:
        return "Medium"
    return "Low"

def prepare_sources(scored_docs):
    sources = []
    for doc, score in scored_docs:
        page = doc.metadata.get("page", "N/A")
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": page + 1 if isinstance(page, int) else page,
            "score": round(float(score), 4),
            "snippet": clean_snippet(doc.page_content, 350)
        })
    return sources

def run_mode(vectorstore, user_input, mode="Q&A", k=4):
    llm = get_llm()
    if llm is None:
        return {"answer": "GOOGLE_API_KEY is missing.", "evidence": "Low", "sources": []}

    scored_docs = retrieve_documents_with_scores(vectorstore, user_input, k=k)

    if not scored_docs:
        return {
            "answer": "I could not find enough evidence in the uploaded documents.",
            "evidence": "Low",
            "sources": []
        }

    evidence = get_evidence_label(scored_docs)
    context = build_context(scored_docs)
    sources = prepare_sources(scored_docs)

    if mode == "Q&A":
        if evidence == "Low":
            return {
                "answer": "I could not find enough evidence in the uploaded documents.",
                "evidence": evidence,
                "sources": sources
            }
        prompt = get_qa_prompt()

    elif mode == "Summary":
        prompt = get_summary_prompt()

    elif mode == "Requirements Extraction":
        prompt = get_requirements_prompt()

    else:
        return {"answer": "Invalid mode selected.", "evidence": "Low", "sources": []}

    chain = prompt | llm
    response = chain.invoke({
        "question": user_input,
        "context": context
    })

    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "evidence": evidence,
        "sources": sources
    }

def compare_documents(vectorstore_a, vectorstore_b, question, k=4):
    llm = get_llm()
    if llm is None:
        return {
            "answer": "GOOGLE_API_KEY is missing.",
            "sources_a": [],
            "sources_b": []
        }

    docs_a = retrieve_documents_with_scores(vectorstore_a, question, k=k)
    docs_b = retrieve_documents_with_scores(vectorstore_b, question, k=k)

    context_a = build_context(docs_a)
    context_b = build_context(docs_b)

    prompt = get_compare_prompt()
    chain = prompt | llm

    response = chain.invoke({
        "question": question,
        "context_a": context_a,
        "context_b": context_b
    })

    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "sources_a": prepare_sources(docs_a),
        "sources_b": prepare_sources(docs_b)
    }