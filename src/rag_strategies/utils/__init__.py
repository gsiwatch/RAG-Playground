"""
Utility functions and helpers for the RAG Strategies package.
"""

from rag_strategies.utils.openai_client import get_llm, get_embeddings_model
from rag_strategies.utils.logger import setup_logger
from rag_strategies.utils.ssl_utils import setup_ssl_certificates

__all__ = [
    'get_llm',
    'get_embeddings_model',
    'setup_logger',
    'setup_ssl_certificates'
]
