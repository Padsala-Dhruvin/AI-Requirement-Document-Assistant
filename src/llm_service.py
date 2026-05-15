import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL
from src.prompt_templates import get_qa_prompt
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
        snippet = clean_snippet(doc.page_content, 800)

        parts.append(
            f"[Chunk {idx}] Source: {source}, Page: {page + 1 if isinstance(page, int) else page}\n"
            f"Content: {snippet}\n"
        )
    return "\n\n".join(parts)

def get_evidence_label(scored_docs):
    """Score evidence based on FAISS L2 distance (lower = better match)."""
    if not scored_docs:
        return "Low"

    best_score = scored_docs[0][1]
    avg_score = sum(score for _, score in scored_docs) / len(scored_docs)

    # Lower FAISS L2 distance = better semantic match
    # Typical good matches are 0.3-0.7, poor matches are > 1.0
    if best_score < 0.5:
        return "High"  # Excellent semantic match
    elif best_score < 0.8:
        return "Medium"  # Good match
    return "Low"  # Weak match

def prepare_sources(scored_docs):
    sources = []

    for doc, score in scored_docs:
        page = doc.metadata.get("page", "N/A")
        source = {
            "source": doc.metadata.get("source", "Unknown"),
            "page": page + 1 if isinstance(page, int) else page,
            "score": round(float(score), 4),
            "snippet": clean_snippet(doc.page_content, 350)
        }
        sources.append(source)

    return sources

def answer_question(vectorstore, question, k=4):
    llm = get_llm()
    if llm is None:
        return {
            "answer": "GOOGLE_API_KEY is missing.",
            "evidence": "Low",
            "sources": []
        }

    scored_docs = retrieve_documents_with_scores(vectorstore, question, k=k)

    if not scored_docs:
        return {
            "answer": "I could not find enough evidence in the uploaded documents.",
            "evidence": "Low",
            "sources": []
        }

    evidence = get_evidence_label(scored_docs)
    context = build_context(scored_docs)

    # Always attempt to generate an answer, regardless of evidence level
    prompt = get_qa_prompt()
    chain = prompt | llm

    try:
        response = chain.invoke({
            "question": question,
            "context": context
        })
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "answer": answer,
        "evidence": evidence,
        "sources": prepare_sources(scored_docs)
    }


def run_mode(vectorstore, user_input, mode="Q&A", k=4):
    """Compatibility wrapper used by app.py for the current UI modes."""
    return answer_question(vectorstore, user_input, k=k)