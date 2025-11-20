from embeddings import get_embeddings, get_embeddings_chain
from config import load_config
from langchain_core.documents import Document


async def main():
    config = load_config()
    document = Document(page_content="Hello, world!")
    splitter = get_embeddings_chain()
    embeddings = get_embeddings(config.embeddings_model_name, config.embeddings_cache_folder)
    chunks = splitter.split_documents([document])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
