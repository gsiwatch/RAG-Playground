from typing import List, Dict
from rag_strategies import get_llm, get_embeddings_model
from rag_strategies.utils.logger import setup_logger
import json
import re

logger = setup_logger(__name__)

class SemanticChunker:
    def __init__(self, llm=None, embeddings=None):
        """Initialize the chunker with LLM and embeddings models"""
        self.llm = llm or get_llm()
        self.embeddings = embeddings or get_embeddings_model()

    async def create_semantic_chunks(self, text: str, document_info: Dict) -> List[Dict]:
        """Create chunks with semantic understanding"""
        try:
            # Create chunks
            chunks = await self._create_chunks(text)
            
            # Enhance chunks with embeddings and metadata
            enhanced_chunks = []
            for chunk in chunks:
                embedding = await self._create_contextual_embedding(
                    chunk['content'],
                    document_info
                )
                root_id = document_info['component_path'].split('_')[0]
                
                metadata = {
                    'component_path': document_info['component_path'],
                    'business_areas': document_info.get('business_areas', []) or [],
                    'products': document_info.get('products', []),
                    'content_type': document_info.get('content_type', 'general'),
                    'applies_to_all_business_areas': document_info.get('applies_to_all_business_areas', False),
                    'chunk_analysis': chunk.get('chunk_metadata', {}),
                    'is_procedure': chunk.get('is_procedure', False),
                    'procedure_type': chunk.get('procedure_type', None),
                    'is_complete_procedure': chunk.get('is_complete_procedure', False)
                }
                
                enhanced_chunks.append({
                    'content': chunk['content'],
                    'metadata': metadata,
                    'embedding': embedding,
                    'root_id': root_id
                })

            if not enhanced_chunks:
                raise ValueError("No valid chunks were created")

            return enhanced_chunks

        except Exception as e:
            logger.error(f"Error in create_semantic_chunks: {str(e)}")
            raise

    async def _create_chunks(self, text: str) -> List[Dict]:
        """Create semantic chunks with metadata"""
        prompt = f"""
        Create semantic chunks from this content following these guidelines:

        1. Content Organization:
        - Keep ALL steps of a procedure together in one chunk
        - Keep definitions with their complete explanations
        - Keep strongly related content together
        - Maintain context within each chunk
        - For procedures, include all related steps in the same chunk
        - For definitions, include the complete definition with any related context

        2. Chunk Properties:
        - Each chunk must be self-contained and meaningful
        - Include necessary context within each chunk
        - Preserve important cross-references
        - Keep chunk size reasonable (200-1000 characters optimal)

        3. CRITICAL - JSON Format Requirements:
        - IMPORTANT: Only use backslashes for JSON escaping (\", \\n) - no other backslashes allowed
        - MOST IMPORTANT: DO NOT escape parentheses () in content text
            - Example CORRECT: "monthly bonus (if applicable)"
            - Example INCORRECT: "monthly bonus $if applicable$"
        - Return ONLY strictly valid JSON chunks
        - Each chunk must be a complete, valid JSON object
        - Properly escape all special characters in JSON strings
        - Use \\n for line breaks in content
        - Do not include any unescaped quotes or special characters
        - Separate chunks with ###
        - Use plain text only - do not include any markdown formatting
        - Never leave properties unquoted
        - Format each chunk EXACTLY as shown:
        {{
            "content": "properly escaped text content",
            "chunk_metadata": {{
                "type": "procedure|definition|overview|general",
                "key_concepts": ["main topics"],
                "relationships": ["related concepts"]
            }},
            "is_procedure": true|false,
            "procedure_type": "string if applicable, null if not",
            "is_complete_procedure": true|false
        }}
        - ABSOLUTELY NO DEVIATIONS from this JSON structure are allowed.
        - ALWAYS verify JSON structure is complete and balanced before returning

        4. Examples of properly formatted chunks:

        Example 1 - Basic Documentation:
        {{
            "content": "Commission Calculation Guidelines:\\n\\nBasic Rate: 5% of total sale\\nMinimum Commission: $100\\n\\nAdjustments:\\n1. Senior Agents: +2%\\n2. Special Programs: +1%\\n3. Training Period: -1%",
            "chunk_metadata": {{
                "type": "procedure",
                "key_concepts": ["commission calculation", "rates"],
                "relationships": ["sales", "compensation"]
            }},
            "is_procedure": true,
            "procedure_type": "calculation",
            "is_complete_procedure": true
        }}

        Example 2 - Policy Information:
        {{
            "content": "Commission Policy Updates for 2024:\\n\\nStandard Commission Structure:\\n- Base Rate: 5%\\n- Volume Bonus: Up to 2%\\n- Referral Bonus: 1%\\n\\nFor complete details, see the Commission Policy Guide.",
            "chunk_metadata": {{
                "type": "policy",
                "key_concepts": ["commission rates", "policy updates"],
                "relationships": ["compensation"]
            }},
            "is_procedure": false,
            "procedure_type": null,
            "is_complete_procedure": false
        }}

        Example 3 - Special Cases:
        {{
            "content": "FHA COVID-19 Impact Guidelines:\\n\\nFor clients affected by COVID-19:\\n1. Review standard qualification criteria\\n2. Document COVID-19 impact\\n3. Apply special consideration factors\\n\\nRefer to full COVID-19 Policy Guide for details.",
            "chunk_metadata": {{
                "type": "procedure",
                "key_concepts": ["FHA", "COVID-19", "qualification"],
                "relationships": ["guidelines", "special consideration"]
            }},
            "is_procedure": true,
            "procedure_type": "qualification",
            "is_complete_procedure": true
        }}

        CRITICAL FORMATTING RULES:
        1. Ensure that all parentheses () are properly balanced and escaped
        2. Convert any HTML or markdown to plain text
        3. Replace links with descriptive text
        4. Remove any special formatting or symbols
        5. Keep content clear and readable
        6. Use proper line breaks (\\n) for formatting
        7. Escape all special characters properly
        8. Remove any raw URLs or IDs
        9. Keep content structure simple and consistent
        10. Maintain proper JSON formatting throughout
        11. Always properly escape dollar signs with a backslash
        12. Never use unescaped backslashes except in valid escape sequences
        13. Ensure all JSON strings are properly quoted and escaped
        14. Remove any HTML entities or special characters
        15. Use only ASCII characters in the output
        16. CRITICAL - For parentheses in content:  
            - Write normal parentheses () directly in the text
            - DO NOT escape normal parentheses
            - Example CORRECT: "Monthly report (includes tax details)"
            - Example INCORRECT: "Monthly report $includes tax details$"
            - Only escape characters that must be escaped in JSON strings (\", \\, \n)

        Example of correct parentheses handling:
        {{
            "content": "Income Calculation Steps:\\n\\n1. Review past records (minimum 2 years)\\n2. Calculate average (excluding outliers)\\n3. Apply adjustment factor \(if required\)",
            "chunk_metadata": {{
                "type": "procedure",
                "key_concepts": ["income calculation"],
                "relationships": ["financial review"]
            }},
            "is_procedure": true,
            "procedure_type": "calculation",
            "is_complete_procedure": true
        }}

        Content to chunk:
        {text}
        """

        response = await self.llm.ainvoke(prompt)
        if not response.content.strip():
            logger.error("Empty response from LLM")
            raise ValueError("Empty response from LLM")

        return self._parse_chunks(response.content)

    def _parse_chunks(self, llm_response: str) -> List[Dict]:
        """Parse LLM response into chunks with metadata"""
        try:
            # Clean and validate response
            clean_response = llm_response.replace('```json', '').replace('```', '').strip()
            raw_chunks = [chunk.strip() for chunk in clean_response.split('###') if chunk.strip()]
            
            if not raw_chunks:
                logger.error("No chunks found in LLM response")
                raise ValueError("No chunks found in LLM response")
            
            chunks = []
            for i, raw_chunk in enumerate(raw_chunks):
                try:
                    # First attempt: try direct JSON parsing
                    logger.debug(f"Character at position 31: '{raw_chunk[31]}'")
                    logger.debug(f"Characters around position 31: '{raw_chunk[25:35]}'")
                    try:
                        chunk_data = json.loads(raw_chunk)
                    except json.JSONDecodeError:
                        # Second attempt: clean the chunk and try again
                        cleaned_chunk = raw_chunk.replace('\n', ' ')
                        cleaned_chunk = re.sub(r'\$?!["\\/bfnrt]|u[0-9a-fA-F]{4})', '', cleaned_chunk)
                        
                        try:
                            chunk_data = json.loads(cleaned_chunk)
                        except json.JSONDecodeError:
                            # If both attempts fail, create a basic chunk structure
                            # Safely extract content using string operations instead of regex
                            content_start = cleaned_chunk.find('"content":"') + 11
                            content_end = cleaned_chunk.find('","chunk_metadata"', content_start)
                            if content_start > 10 and content_end > content_start:
                                content = cleaned_chunk[content_start:content_end]
                            else:
                                content = "Error: Could not parse content"

                            # Create a basic chunk structure
                            chunk_data = {
                                "content": content,
                                "chunk_metadata": {
                                    "type": "general",
                                    "key_concepts": [],
                                    "relationships": []
                                },
                                "is_procedure": False,
                                "procedure_type": None,
                                "is_complete_procedure": False
                            }
                    
                    # Validate required fields
                    if not all(k in chunk_data for k in ['content', 'chunk_metadata']):
                        logger.error(f"Chunk {i + 1} missing required fields")
                        raise ValueError(f"Chunk {i + 1} format invalid: missing required fields")
                    
                    # Clean up the content field
                    if isinstance(chunk_data['content'], str):
                        chunk_data['content'] = chunk_data['content'].replace('\\n', '\n')
                    
                    chunks.append(chunk_data)
                    
                except Exception as e:
                    logger.error(f"Failed to parse chunk {i + 1}: {str(e)}")
                    logger.error(f"Problematic chunk: {raw_chunk[:200]}...")
                    logger.debug(f"Full problematic chunk: {raw_chunk}")
                    raise
            
            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
                
        except Exception as e:
            logger.error(f"Error in _parse_chunks: {str(e)}")
            raise
    
    async def _create_contextual_embedding(self, content: str, document_info: Dict) -> List[float]:
        """Create embedding with document context"""
        try:
            # Handle None values for document info fields
            business_areas = document_info.get('business_areas', [])
            if business_areas is None:
                business_areas = []
                
            products = document_info.get('products', [])
            if products is None:
                products = []
                
            tags = document_info.get('tags', [])
            if tags is None:
                tags = []

            context = f"""
            Document: {document_info.get('page_title', '')}
            Content Type: {document_info.get('content_type', 'general')}
            Business Areas: {', '.join(business_areas)}
            Products: {', '.join(products)}
            
            Content: {content}
            """
            
            return await self.embeddings.embed_query(context)
            
        except Exception as e:
            logger.error(f"Error creating contextual embedding: {str(e)}")
            logger.error(f"Document info: {document_info}")
            raise
