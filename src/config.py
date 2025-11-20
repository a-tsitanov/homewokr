from dataclasses import dataclass

from environs import Env


@dataclass
class Config:
    embeddings_model_name: str
    embeddings_cache_folder: str


def load_config():
    env = Env()
    env.read_env()
    return Config(
        embeddings_model_name=env.str(
            "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embeddings_cache_folder=env.str("EMBEDDINGS_CACHE_FOLDER", "./.cache"),
    )
