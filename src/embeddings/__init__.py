from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from huggingface_hub import snapshot_download
from pathlib import Path
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"


def prepare_model(model_name: str, cache_root: str, model_type: str = "auto") -> str:
    """
    Скачивает модель в отдельную подпапку cache_root/<model_name>/
    model_type = "llm" | "embedding" | "sentence-transformers" | "auto"
    Возвращает путь к локальной модели для использования оффлайн.
    """

    safe_name = model_name.replace("/", "_")
    model_dir = Path(cache_root) / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем, есть ли что-то уже в папке
    if any(model_dir.iterdir()):
        print(f"[INFO] Модель {model_name} уже загружена в {model_dir}")
        return str(model_dir)

    # Для sentence-transformers используем SentenceTransformer.save
    if model_type == "sentence-transformers" or (model_type == "auto" and "sentence-transformers" in model_name):
        print(f"[INFO] Скачиваю sentence-transformers модель {model_name} …")
        model = SentenceTransformer(model_name)
        model.save(str(model_dir))
        print(f"[INFO] Модель сохранена в {model_dir}")
        return str(model_dir)

    # Для обычных HF моделей
    print(f"[INFO] Скачиваю HF модель {model_name} …")
    snapshot_dir = snapshot_download(
        repo_id=model_name,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False
    )
    print(f"[INFO] Модель сохранена в {snapshot_dir}")
    return snapshot_dir


def get_embeddings(model_name: str, cache_folder: str):

    
    return HuggingFaceEmbeddings(
        model_name="./.models/sentence-transformers_all_minilm_l6_v2",
        model_kwargs={"local_files_only": True}
    )

def get_embeddings_chain():
    return RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
