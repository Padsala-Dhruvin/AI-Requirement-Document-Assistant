from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import EMBED_MODEL

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)