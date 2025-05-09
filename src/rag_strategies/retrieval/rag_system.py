from typing import List, Dict, Optional
from datetime import datetime

from rag_strategies.retrieval.query_processor import QueryProcessor
from rag_strategies.retrieval.response_generator import ResponseGenerator
from rag_strategies.utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGSystem:
    def __init__(self):
        self.query_processor = None
        self.response_generator = None

    @classmethod
    async def create(cls):
        """Factory method to create instance"""
        self = cls()
        try:
            self.query_processor = await QueryProcessor.create()
            self.response_generator = await ResponseGenerator.create()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize RAG System: {str(e)}")
            raise

    async def process_query(
        self,
        query: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Process query and generate response"""
        try:
            # 1. Search in summaries and get relevant chunks
            search_results = await self.query_processor.search(query)
            
            # 2. Generate response
            response = await self.response_generator.generate_response(
                query=query,
                search_result=search_results,
                metadata=metadata
            )

            # 3. Add basic processing info
            response['metadata'] = {
                **response.get('metadata', {}),
                'processed_at': datetime.utcnow().isoformat(),
                'sources_used': {
                    'summaries': len(search_results.get('summaries', [])),
                    'chunks': len(search_results.get('chunks', []))
                }
            }

            return response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your question.",
                "citations": [],
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    async def process_batch_queries(
        self,
        queries: List[str],
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Process multiple queries"""
        responses = []
        
        for query in queries:
            try:
                response = await self.process_query(
                    query=query,
                    metadata=metadata
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                responses.append({
                    "error": str(e),
                    "query": query
                })

        return responses

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.query_processor:
                await self.query_processor.cleanup()
            if self.response_generator:
                await self.response_generator.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
