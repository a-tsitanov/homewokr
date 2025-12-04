# 1. Установка (один раз)
# pip install langchain langchain-experimental langchain-huggingface sentence-transformers spacy
# python -m spacy download ru_core_news_sm

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SpacyTextSplitter
# from langchain_chroma import Chroma
from langchain_core.documents import Document
import re

# ===================================================================
# 2. Самые лучшие настройки на 2025 год для русского языка
# ===================================================================

# Модель №1 по качеству на русском (MTEB-ru лидер)
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",   # или BAAI/bge-m3
    # model_kwargs={"device": "cuda"},                        # "cpu" если нет GPU
    encode_kwargs={"normalize_embeddings": True}            # ОБЯЗАТЕЛЬНО!
)

# Семантический чанкер — определяет смену темы
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",   # самый стабильный и адаптивный
    breakpoint_threshold_amount=94,           # 92–96 — золотая середина
                                              # 94 = темы меняются чётко, но не слишком мелко
)

# Резервный аккуратный сплиттер (если семантический чанк слишком большой)
spacy_splitter = SpacyTextSplitter(
    chunk_size=600,          # символов
    chunk_overlap=100,
    separator="\n",
    language="ru_core_news_sm"
)

# ===================================================================
# 3. Функция, которую ты будешь вызывать на каждый новый транскрипт
# ===================================================================

def create_chunks_from_transcript(transcript: str, call_id: str = "unknown"):
    # 3.1 Небольшая предобработка
    text = re.sub(r'\n{3,}', '\n\n', transcript.strip())  # убираем лишние пустые строки
    text = re.sub(r' +', ' ', text)                       # множественные пробелы → один

    # 3.2 Сначала грубые семантические чанки (главное!)
    rough_chunks = semantic_chunker.split_text(text)

    # 3.3 Каждый грубый чанк при необходимости дорезаем аккуратно по предложениям
    final_chunks = []
    for i, chunk in enumerate(rough_chunks):
        # Если чанк слишком большой (> 1000 символов) — дорезаем Spacy
        if len(chunk) > 1000:
            subchunks = spacy_splitter.split_text(chunk)
            final_chunks.extend(subchunks)
        else:
            final_chunks.append(chunk)

    # 3.4 Оборачиваем в Document с метаданными (очень важно для фильтрации!)
    documents = []
    for i, chunk in enumerate(final_chunks):
        documents.append(Document(
            page_content=chunk.strip(),
            metadata={
                "call_id": call_id,
                "chunk_id": i,
                "source": "transcript",
                "char_start": text.find(chunk),           # приблизительно
                "char_end": text.find(chunk) + len(chunk),
            }
        ))

    return documents

# ===================================================================
# 4. Пример использования
# ===================================================================

transcript = """
Спикер A: Добрый день, Анна! Как вы узнали о вакансии?
Спикер B: Через HH, подруга скинула.
Спикер A: Расскажите про опыт в продажах сложных решений.
Спикер B: 4 года в IT-продажах, средний чек 5+ млн, работала с банками.
Спикер A: Лучший месяц?
Спикер B: 47 миллионов в прошлом году.
Спикер A: Отлично! Теперь условия: оклад 120 на руки, 2% от личных продаж + бонусы.
Спикер B: Оклад маловат, хочу 250+.
Спикер A: Готовы к командировкам 40% времени?
Спикер B: Да, привыкла.
Спикер A: Спасибо, свяжемся через 3 дня.
"""

docs = create_chunks_from_transcript(transcript, call_id="call_2025_001")

print(f"Получено чанков: {len(docs)}\n")
for d in docs:
    print(f"[{d.metadata['chunk_id']}] {d.page_content[:120]}...\n")

# ===================================================================
# 5. Сохраняем в векторную базу (один раз или инкрементально)
# ===================================================================

# vectorstore = Chroma.from_documents(
#     documents=docs,
#     embedding=embeddings,
#     collection_name="transcripts",
#     persist_directory="./chroma_db"   # или подключение к Qdrant/Pinecone/Milvus
# )

# # ===================================================================
# # 6. Теперь поиск по запросу пользователя (это и есть RAG)
# # ===================================================================

# query = "Где обсуждали зарплату и ожидания кандидата?"

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
# relevant_chunks = retriever.invoke(query)

# print("Найденные фрагменты:")
# for chunk in relevant_chunks:
#     print("— " * 30)
#     print(chunk.page_content)