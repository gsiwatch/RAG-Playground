# TODO- update how we calculate confidence score by using cosine similarity 
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from rag_strategies import get_llm, get_embeddings_model
from rag_strategies.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Citation:
    """Structure for holding citation information"""
    content: str
    document_path: str
    metadata: Dict
    confidence: float

class ResponseGenerator:
    def __init__(self):
        self.llm = None
        self.embeddings = None

    @classmethod
    async def create(cls):
        """Factory method to create instance"""
        self = cls()
        try:
            self.llm = get_llm()
            self.embeddings = get_embeddings_model()
            return self
        except Exception as e:
            logger.error(f"Failed to initialize ResponseGenerator: {str(e)}")
            raise

    async def generate_response(
        self,
        query: str,
        search_result: Dict,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Generate response using search results"""
        try:
            if not search_result.get('chunks') and not search_result.get('summaries'):
                return self._create_no_content_response()

            if search_result.get('summaries'):
                logger.info("Generating response from summaries and chunks")
                response = await self._generate_comprehensive_response(
                    query,
                    search_result,
                    metadata
                )
            else:
                logger.info("Generating response from chunks only")
                response = await self._generate_chunks_response(
                    query,
                    search_result,
                    metadata
                )

            response['metadata'].update({
                'processed_at': datetime.utcnow().isoformat(),
                'search_type': search_result.get('metadata', {}).get('search_type', 'unknown'),
                **(metadata or {})
            })

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._create_error_response(str(e))

    async def _generate_comprehensive_response(
        self,
        query: str,
        search_result: Dict,
        metadata: Optional[Dict]
    ) -> Dict:
        """Generate response using both summaries and chunks"""
        try:
            summaries = search_result['summaries']
            chunks = search_result['chunks']

            context = self._create_comprehensive_context(summaries, chunks)

            prompt = f"""
            Answer the following question using the provided context. 
            
            Question: {query}

            Context:
            {context}

            Requirements:
            1. Answer directly and concisely - aim for 3-4 sentences unless more detail is absolutely necessary
            2. Use only information from the provided context
            3. If multiple sources provide different information, note the differences briefly
            4. If the information is incomplete, acknowledge what's missing
            5. Use simple text formatting without markdown, bullets, or special characters
            6. Start with a direct answer, then provide necessary context or conditions
            7. Do not use headers, sections, or extensive formatting
            8. Do not use bold, italics, or other markdown formatting
            9. Keep the response focused and to the point
            10. If a longer response is needed, organize it in clear paragraphs

            Example of good response format:
            The maximum debt-to-income ratio for conventional loans is typically 45%, but it can go up to 50% with strong compensating factors such as excellent credit score and significant reserves. Different loan types may have varying requirements, with FHA allowing up to 57% in some cases. The specific limit depends on the loan program, property type, and borrower's overall financial profile.

            Answer:
            """

            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            citations = self._create_citations(summaries, chunks)

            return {
                "answer": answer,
                "citations": citations,
                "metadata": {
                    "sources_used": {
                        "summaries": len(summaries),
                        "chunks": len(chunks)
                    }
                },
                "confidence": self._calculate_confidence(citations, summaries)
            }

        except Exception as e:
            logger.error(f"Error in comprehensive response generation: {str(e)}")
            raise

    async def _generate_chunks_response(
        self,
        query: str,
        search_result: Dict,
        metadata: Optional[Dict]
    ) -> Dict:
        """Generate response using only chunks"""
        try:
            chunks = search_result['chunks']
            
            prompt = f"""
            Answer the following question using only the provided information.
            
            Question: {query}

            Information:
            {self._format_chunks_for_prompt(chunks)}

            Requirements:
            1. Use only the information from the provided content
            2. Be concise and direct - aim for 3-4 sentences unless more detail is absolutely necessary
            3. If information is missing or unclear, state that explicitly
            4. Use simple text formatting without markdown, bullets, or special characters
            5. If there are conditions or exceptions, include them briefly in the main response
            6. Start with a direct answer, then provide necessary context or conditions
            7. Do not use headers, sections, or extensive formatting
            8. Do not use bold, italics, or other markdown formatting
            9. Keep the response focused and to the point
            10. If a longer response is needed, organize it in clear paragraphs

            Example of good response format:
            A property can be considered a primary residence if the client occupies it as their main home within 60 days of closing and intends to live there. The client must not own or occupy another primary residence at the same time. Special provisions exist for military personnel and remote workers, where occupancy by a spouse or family member may be permitted with proper documentation.

            Answer:
            """

            response = await self.llm.ainvoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            citations = self._create_chunk_citations(chunks)

            return {
                "answer": answer,
                "citations": citations,
                "metadata": {
                    "sources_used": {
                        "chunks": len(chunks)
                    }
                },
                "confidence": self._calculate_chunk_confidence(citations)
            }

        except Exception as e:
            logger.error(f"Error in chunks response generation: {str(e)}")
            raise

    def _create_comprehensive_context(
        self,
        summaries: List[Dict],
        chunks: List[Dict]
    ) -> str:
        """Create context from summaries and chunks"""
        context_parts = []

        # Add summary context
        for summary in summaries:
            context_parts.append(f"""
            Summary:
            {summary.get('summary', '')}
            """)

        # Add relevant chunks
        for chunk in chunks:
            context_parts.append(f"""
            Detail:
            {chunk.get('content', '')}
            """)

        return "\n\n".join(context_parts)

    def _create_citations(
        self,
        summaries: List[Dict],
        chunks: List[Dict]
    ) -> List[Dict]:
        """Create citations from summaries and chunks"""
        citations = []

        # Add summary citations
        for summary in summaries:
            citations.append({
                "content": summary.get('summary', ''),
                "metadata": {
                    "document_path": "summary",
                    "metadata": summary.get('metadata', {}),
                    # TODO - add the code to use cosine similarity to get the confidence score based on semantic, relevance, similarity etc.
                    # instead of hard coded value
                    "confidence": 0.9 
                }
            })

        citations.extend(self._create_chunk_citations(chunks))

        return citations

    def _create_chunk_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Create citations from chunks"""
        return [
            {
                "content": chunk.get('content', ''),
                "metadata": {
                    "document_path": chunk.get('metadata', {}).get('component_path', ''),
                    "metadata": chunk.get('metadata', {}),
                    "confidence": chunk.get('score', 0.7)
                }
            }
            for chunk in chunks
        ]

    def _calculate_confidence(
        self,
        citations: List[Dict],
        summaries: List[Dict]
    ) -> float:
        """Calculate overall confidence score"""
        if not citations:
            return 0.0

        scores = []
        
        # Consider summary presence
        if summaries:
            scores.append(0.9)
        
        # Consider citation scores
        citation_scores = [cit['metadata']['confidence'] for cit in citations]
        if citation_scores:
            scores.append(sum(citation_scores) / len(citation_scores))

        return min(0.95, sum(scores) / len(scores))

    def _calculate_chunk_confidence(self, citations: List[Dict]) -> float:
        """Calculate confidence for chunk-based response"""
        if not citations:
            return 0.0

        scores = [cit['metadata']['confidence'] for cit in citations]
        return min(0.95, sum(scores) / len(scores))

    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Format chunks for prompt"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            formatted_chunks.append(f"""
            Source {i}:
            {chunk.get('content', '')}
            """)
            
        return "\n".join(formatted_chunks)

    def _create_no_content_response(self) -> Dict:
        """Create response for no content found"""
        return {
            "answer": "I apologize, but I couldn't find relevant information to answer your question accurately.",
            "citations": [],
            "metadata": {
                "error": "No relevant content found"
            },
            "confidence": 0.0
        }

    def _create_error_response(self, error: str) -> Dict:
        """Create response for error cases"""
        return {
            "answer": "I apologize, but I encountered an error while processing your question.",
            "citations": [],
            "metadata": {
                "error": error,
                "error_time": datetime.utcnow().isoformat()
            },
            "confidence": 0.0
        }

    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Add any cleanup needed for ResponseGenerator
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
