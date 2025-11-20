from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_embeddings(model_name: str, cache_folder: str):
    return HuggingFaceEmbeddings(model_name=model_name, cache_folder=cache_folder)

def get_embeddings_chain():
    return RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
