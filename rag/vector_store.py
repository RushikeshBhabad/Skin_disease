"""
FAISS vector store for the RAG system.
Uses HuggingFace Inference API embeddings — no local model downloads.
"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from rag.medical_knowledge import get_medical_documents
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache for the vector store
_vector_store_cache: Optional[FAISS] = None


def _get_embeddings(config: Config) -> HuggingFaceEndpointEmbeddings:
    """
    Create a HuggingFace Inference API embeddings instance.

    Args:
        config: Application config with HuggingFace API key.

    Returns:
        Embeddings model instance using the cloud API.
    """
    if not config.has_huggingface:
        raise ValueError(
            "HuggingFace API key is required for embeddings. "
            "Please set HUGGINGFACE_API_KEY in .env or enter it in the sidebar."
        )

    return HuggingFaceEndpointEmbeddings(
        model=config.embedding_model,
        huggingfacehub_api_token=config.huggingface_api_key,
    )


def build_vector_store(config: Config, force_rebuild: bool = False) -> FAISS:
    """
    Build or retrieve a cached FAISS vector store from medical knowledge.

    Args:
        config: Application config.
        force_rebuild: If True, rebuild even if a cached store exists.

    Returns:
        FAISS vector store ready for similarity search.
    """
    global _vector_store_cache

    if _vector_store_cache is not None and not force_rebuild:
        logger.info("Returning cached vector store")
        return _vector_store_cache

    logger.info("Building FAISS vector store from medical knowledge corpus...")

    embeddings = _get_embeddings(config)
    raw_docs = get_medical_documents()

    # Convert to LangChain Document objects
    documents = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"],
        )
        for doc in raw_docs
    ]

    logger.info(f"Embedding {len(documents)} medical knowledge chunks...")

    vector_store = FAISS.from_documents(documents, embeddings)
    _vector_store_cache = vector_store

    logger.info("FAISS vector store built successfully")
    return vector_store


def get_retriever(config: Config, top_k: Optional[int] = None):
    """
    Get a LangChain retriever backed by the FAISS vector store.

    Args:
        config: Application config.
        top_k: Number of documents to retrieve. Defaults to config.rag_top_k.

    Returns:
        LangChain retriever instance.
    """
    k = top_k or config.rag_top_k
    store = build_vector_store(config)
    return store.as_retriever(search_kwargs={"k": k})
