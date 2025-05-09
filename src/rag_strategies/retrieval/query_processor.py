from typing import List, Dict
from motor.motor_asyncio import AsyncIOMotorClient

from rag_strategies.config import settings
from rag_strategies.utils.logger import setup_logger
from rag_strategies.utils.openai_client import get_embeddings_model, get_llm

logger = setup_logger(__name__)

class QueryProcessor:
    def __init__(self):
        """Initialize QueryProcessor with database connections and models"""
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db_name]
        self.summary_collection = self.db[settings.mongodb_summary_collection]
        self.chunks_collection = self.db[settings.mongodb_chunks_collection]
        self.embeddings = get_embeddings_model()
        self.llm = get_llm()

    @classmethod
    async def create(cls):
        """Factory method to create instance"""
        try:
            self = cls()
            # Verify MongoDB connection
            await self.client.admin.command('ping')
            logger.info("MongoDB connection verified")
            return self
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()

    async def search(self, query: str) -> Dict:
        """Execute search with summary-first approach"""
        try:
            # Query embedding
            query_embedding = await self.embeddings.embed_query(query)
            
            # 1. Search in summaries
            relevant_summaries = await self._search_summaries(query_embedding)
            logger.info(f"Found {len(relevant_summaries)} relevant summaries")

            # 2. Get chunks from relevant summaries
            chunks_from_summaries = await self._get_chunks_from_summaries(
                relevant_summaries,
                query_embedding
            )
            logger.info(f"Found {len(chunks_from_summaries)} chunks from summaries/related content")

            chunks = chunks_from_summaries
            if len(chunks) < 3: 
                additional_chunks = await self._direct_chunk_search(
                    query_embedding,
                    existing_chunk_ids=[str(c['_id']) for c in chunks]
                )
                chunks.extend(additional_chunks)
                logger.info(f"Added {len(additional_chunks)} additional chunks")

            processed_chunks = self._process_chunks(chunks)

            return {
                'chunks': processed_chunks,
                'summaries': relevant_summaries,
                'metadata': {
                    'total_chunks': len(processed_chunks),
                    'summary_count': len(relevant_summaries),
                    'search_type': 'hybrid'
                }
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    async def _search_summaries(self, query_embedding: List[float]) -> List[Dict]:
        """Search in summary collection using vector similarity"""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "complied_answers_testing_summary_embeddings",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "summary_embedding",
                            "k": 5
                        }
                    }
                },
                {
                    "$project": {
                        "root_id": 1,
                        "page_title": 1,
                        "summary": 1,
                        "source_documents": 1,
                        "metadata": 1,
                        "score": { "$meta": "searchScore" }
                    }
                }
            ]

            results = await self.summary_collection.aggregate(pipeline).to_list(None)
            
            # Log summary results
            logger.info(f"Found {len(results)} summaries from vector search")
            for idx, summary in enumerate(results):
                logger.info(f"Summary {idx + 1}:")
                logger.info(f"  Title: {summary.get('page_title', 'No title')}")
                logger.info(f"  Score: {summary.get('score', 'No score')}")
                logger.info(f"  Summary: {summary.get('summary', 'No summary')[:200]}...")

            return results

        except Exception as e:
            logger.error(f"Summary search failed: {str(e)}")
            return []

    async def _get_chunks_from_summaries(
        self,
        summaries: List[Dict],
        query_embedding: List[float]
    ) -> List[Dict]:
        """Get relevant chunks from summaries"""
        try:
            if not summaries:
                return []

            summary_ids = [summary['_id'] for summary in summaries]
            root_ids = [summary.get('root_id') for summary in summaries if summary.get('root_id')]

            logger.info(f"Searching chunks with {len(summary_ids)} summary IDs and {len(root_ids)} root IDs")

            pipeline = [
                {
                    "$search": {
                        "index": "complied_answers_test_chunks_embeddings",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": 10
                        }
                    }
                },
                {
                    "$match": {
                        "$or": [
                            {"summary_id": {"$in": summary_ids}},
                            {"root_id": {"$in": root_ids}},
                        ]
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "root_id": 1,
                        "summary_id": 1,
                        "score": { "$meta": "searchScore" }
                    }
                },
                {
                    "$limit": 10
                }
            ]

            results = await self.chunks_collection.aggregate(pipeline).to_list(None)
            
            if results:
                logger.info(f"Found {len(results)} chunks from related summaries")
                for idx, chunk in enumerate(results):
                    logger.info(f"Chunk {idx + 1}:")
                    logger.info(f"  Score: {chunk.get('score', 'No score')}")
                    logger.info(f"  Content: {chunk.get('content', 'No content')[:200]}...")
                    logger.info(f"  Summary ID: {chunk.get('summary_id', 'No summary ID')}")
                    logger.info(f"  Root ID: {chunk.get('root_id', 'No root ID')}")
            else:
                logger.info("No chunks found from related summaries, falling back to direct search")
                pipeline = [
                    {
                        "$search": {
                            "index": "complied_answers_test_chunks_embeddings",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "embedding",
                                "k": 5
                            }
                        }
                    },
                    {
                        "$project": {
                            "content": 1,
                            "metadata": 1,
                            "root_id": 1,
                            "summary_id": 1,
                            "score": { "$meta": "searchScore" }
                        }
                    }
                ]
                results = await self.chunks_collection.aggregate(pipeline).to_list(None)
                
                logger.info(f"Found {len(results)} chunks from fallback search")
                for idx, chunk in enumerate(results):
                    logger.info(f"Fallback Chunk {idx + 1}:")
                    logger.info(f"  Score: {chunk.get('score', 'No score')}")
                    logger.info(f"  Content: {chunk.get('content', 'No content')[:200]}...")
                    logger.info(f"  Summary ID: {chunk.get('summary_id', 'No summary ID')}")
                    logger.info(f"  Root ID: {chunk.get('root_id', 'No root ID')}")

            return results

        except Exception as e:
            logger.error(f"Error getting chunks from summaries: {str(e)}")
            return []

    async def _direct_chunk_search(
        self,
        query_embedding: List[float],
        existing_chunk_ids: List[str]
    ) -> List[Dict]:
        """Direct search in chunks collection"""
        try:
            pipeline = [
                {
                    "$search": {
                        "index": "complied_answers_test_chunks_embeddings",
                        "knnBeta": {
                            "vector": query_embedding,
                            "path": "embedding",
                            "k": 10
                        }
                    }
                },
                {
                    "$match": {
                        "_id": { "$nin": existing_chunk_ids }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "root_id": 1,
                        "summary_id": 1,
                        "score": { "$meta": "searchScore" },
                        "_id": 1
                    }
                },
                {
                    "$limit": 5
                }
            ]

            results = await self.chunks_collection.aggregate(pipeline).to_list(None)
            
            logger.info(f"Found {len(results)} additional chunks from direct search")
            for idx, chunk in enumerate(results):
                logger.info(f"Additional Chunk {idx + 1}:")
                logger.info(f"  Score: {chunk.get('score', 'No score')}")
                logger.info(f"  Content: {chunk.get('content', 'No content')[:200]}...")
                logger.info(f"  Summary ID: {chunk.get('summary_id', 'No summary ID')}")
                logger.info(f"  Root ID: {chunk.get('root_id', 'No root ID')}")

            return results

        except Exception as e:
            logger.error(f"Direct chunk search failed: {str(e)}")
            return []

    def _process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Process and deduplicate chunks"""
        seen_content = set()
        processed_chunks = []
        
        for chunk in chunks:
            normalized_content = self._normalize_content(chunk['content'])
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                
                processed_chunk = {
                    "content": chunk['content'],
                    "metadata": chunk.get('metadata', {}),
                    "root_id": chunk.get('root_id'),
                    "summary_id": chunk.get('summary_id'),
                    "score": chunk.get('score', 0.7)
                }
                
                processed_chunks.append(processed_chunk)
        
        # Sort by score
        processed_chunks.sort(key=lambda x: x['score'], reverse=True)
        return processed_chunks

    def _normalize_content(self, content: str) -> str:
        """Normalize content for deduplication"""
        return ' '.join(content.lower().split())
