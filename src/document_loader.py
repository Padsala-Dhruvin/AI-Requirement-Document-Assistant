from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

def load_pdf_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = doc.metadata.get("page", i)
            doc.metadata["chunk_id"] = f"{uploaded_file.name}_page_{doc.metadata['page'] + 1}"

        documents.extend(docs)
        os.remove(tmp_path)

    return documents


def load_pdf_documents_grouped(uploaded_files):
    grouped_docs = {}

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = doc.metadata.get("page", i)
            doc.metadata["chunk_id"] = f"{uploaded_file.name}_page_{doc.metadata['page'] + 1}"

        grouped_docs[uploaded_file.name] = docs
        os.remove(tmp_path)

    return grouped_docs

def load_pdf_documents_grouped(uploaded_files):
    grouped_docs = {}

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for i, doc in enumerate(docs):
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["page"] = doc.metadata.get("page", i)
            doc.metadata["chunk_id"] = f"{uploaded_file.name}_page_{doc.metadata['page'] + 1}"

        grouped_docs[uploaded_file.name] = docs
        os.remove(tmp_path)

    return grouped_docs