"""
RAG Strategies package for question answering using documents.
"""
from rag_strategies.utils.logger import setup_logger
from rag_strategies.config import settings
from rag_strategies.utils.openai_client import get_llm, get_embeddings_model

__version__ = "0.1.0"

logger = setup_logger(__name__)

__all__ = [
    'settings',
    'setup_logger',
    'get_llm',
    'get_embeddings_model',
    'logger',
]

logger.info(f"Initialized {__name__} version {__version__}")
