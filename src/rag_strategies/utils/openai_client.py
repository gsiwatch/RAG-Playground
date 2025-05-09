from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_strategies.config import settings
from rag_strategies.utils.logger import setup_logger
from rag_strategies.utils.ssl_utils import setup_ssl_certificates
from threading import Lock
from typing import List

logging = setup_logger(__name__)
setup_ssl_certificates()

class AsyncOpenAIEmbeddings(OpenAIEmbeddings):
    """Async wrapper for OpenAI embeddings"""
    
    async def embed_query(self, text: str) -> List[float]:
        """Async embedding for single text"""
        try:
            embeddings = await super().aembed_query(text)
            return embeddings
        except Exception as e:
            logging.error(f"Error in embed_query: {str(e)}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embedding for multiple texts"""
        try:
            embeddings = await super().aembed_documents(texts)
            return embeddings
        except Exception as e:
            logging.error(f"Error in embed_documents: {str(e)}")
            raise

class OpenAIClient:
    _instance = None
    _lock = Lock()
    _initialized = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logging.info("Initializing OpenAI client")
            self._embeddings_model = AsyncOpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )
            self._llm = ChatOpenAI(
                model_name=settings.openai_model_name,
                temperature=0.1,
                openai_api_key=settings.openai_api_key,
                max_tokens=4096,
                request_timeout=120,
                top_p=0.95,
                presence_penalty=0,
                frequency_penalty=0,
                streaming=False
            )
            self._initialized = True
    
    @property
    def embeddings(self) -> AsyncOpenAIEmbeddings:
        return self._embeddings_model
    
    @property
    def llm(self) -> ChatOpenAI:
        return self._llm

# Create single instance
_client = OpenAIClient()

# Public interface
def get_embeddings_model() -> AsyncOpenAIEmbeddings:
    return _client.embeddings

def get_llm() -> ChatOpenAI:
    return _client.llm
