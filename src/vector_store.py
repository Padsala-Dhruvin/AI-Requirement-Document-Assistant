from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model


def create_vectorstore(chunks):
    embedding_model = get_embedding_model()
    return FAISS.from_documents(chunks, embedding_model)


def create_vectorstores_for_documents(grouped_chunks):
    embedding_model = get_embedding_model()
    stores = {}

    for filename, chunks in grouped_chunks.items():
        stores[filename] = FAISS.from_documents(chunks, embedding_model)

    return stores


def retrieve_documents(vectorstore, query, k=4):
    return vectorstore.similarity_search(query, k=k)


def retrieve_documents_with_scores(vectorstore, query, k=4):
    return vectorstore.similarity_search_with_score(query, k=k)
