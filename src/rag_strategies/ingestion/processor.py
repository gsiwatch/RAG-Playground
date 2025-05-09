from datetime import datetime
from typing import List, Dict
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from rag_strategies.ingestion.cleaner import ContentCleaner
from rag_strategies.ingestion.chunker import SemanticChunker
from rag_strategies import get_embeddings_model, get_llm, setup_logger, settings

logger = setup_logger(__name__)

class DocumentProcessor:    
    async def __ainit__(self):
        """Async initialization"""
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db_name]
        self.docs_collection = self.db[settings.mongodb_collection_name]
        self.summary_collection = self.db[settings.mongodb_summary_collection]
        self.chunks_collection = self.db[settings.mongodb_chunks_collection]
        
        self.chunker = SemanticChunker()
        self.embeddings = get_embeddings_model()
        self.llm = get_llm()
        self.cleaner = ContentCleaner()
        
        return self
    
    @classmethod
    async def create(cls):
        """Factory method to create instance"""
        self = cls()
        await self.__ainit__()
        return self

    async def cleanup(self): 
        """Cleanup resources"""
        try:
            if self.client:
                self.client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def process_root_documents(self, root_id: str):
        """Process all documents for a given root_id"""
        try:
            stats = {
                "total_documents": 0,
                "total_chunks": 0,
                "chunks_by_type": {},
                "average_chunk_size": 0,
                "processing_start": datetime.utcnow()
            }

            # Get all related documents
            related_docs = await self._get_related_documents(root_id)
            if not related_docs:
                logger.warning(f"No documents found for root_id: {root_id}")
                return

            stats["total_documents"] = len(related_docs)

            # Clean all documents content
            for doc in related_docs:
                doc['Content'] = self.cleaner.clean_content(doc['Content'])

            # Create summary
            summary_doc = await self._create_summary(related_docs)
            summary_id = await self._store_summary(summary_doc)
            
            # Process each document for chunks
            chunk_sizes = []
            for doc in related_docs:
                chunks = await self._process_document(doc, summary_id)
                if chunks:  # Only process if chunks were returned
                    stats["total_chunks"] += len(chunks)
                    chunk_sizes.extend([len(c['content']) for c in chunks])
                    
                    # Count chunk types
                    for chunk in chunks:
                        chunk_type = chunk['metadata']['chunk_analysis']['type']
                        stats["chunks_by_type"][chunk_type] = stats["chunks_by_type"].get(chunk_type, 0) + 1

            # Just for logging
            if chunk_sizes:
                stats["average_chunk_size"] = sum(chunk_sizes) / len(chunk_sizes)
            
            stats["processing_end"] = datetime.utcnow()
            stats["processing_time"] = (stats["processing_end"] - stats["processing_start"]).total_seconds()

            logger.info(f"Successfully processed root_id: {root_id}")
            logger.info(f"Processing statistics: {stats}")

        except Exception as e:
            logger.error(f"Error processing root_id {root_id}: {str(e)}")
            raise

    async def _get_related_documents(self, root_id: str) -> List[Dict]:
        """Get all documents related to root_id with specific field exclusions"""
        query = {
            "ComponentPath": {"$regex": f"^{root_id}_.*"}
        }
        projection = {
            "ContentEmbeddings": 0,
            "IsEmbeddingsTruncated": 0,
            "EmbeddingsChunkId": 0
        }
        
        return await self.docs_collection.find(
            query,
            projection
        ).sort("ComponentPath", 1).to_list(None)

    async def _create_summary(self, documents: List[Dict]) -> Dict:
        """Create summary using all documents content"""
        try:
            logger.info(f"Creating summary for {len(documents)} documents")
            
            # Combine all documents content
            combined_content = "\n\n".join(doc["Content"] for doc in documents)
            
            prompt = f"""
            Create a comprehensive summary of these related documents following these guidelines:
            1. Begin with a clear overview
            2. Identify main topics and key concepts
            3. Highlight important relationships between documents
            4. Include key steps or procedures if present
            5. Note any important definitions or terms
            6. Maintain context and relationships
            7. Structure the summary with clear sections

            Content:
            {combined_content}

            Provide a well-structured summary that captures the main points and relationships.
            """

            logger.debug("Sending summary request to LLM")
            response = await self.llm.ainvoke(prompt)
            summary = response.content
            
            logger.debug(f"Summary generated: {len(summary)} characters")
            
            # Create summary embedding
            summary_embedding = await self._create_summary_embedding(summary, documents[0])
            
            main_doc = documents[0]
            business_areas = main_doc.get("BusinessAreas", []) 
            
            summary_doc = {
                "root_id": self._extract_root_id(main_doc["ComponentPath"]),
                "page_title": main_doc["PageTitle"],
                "summary": summary,
                "summary_embedding": summary_embedding,
                "source_documents": [
                    {
                        "doc_id": str(doc["_id"]),
                        "component_path": doc["ComponentPath"],
                        "page_title": doc["PageTitle"]
                    }
                    for doc in documents
                ],
                "metadata": {
                    "channels": main_doc.get("Channels", []),
                    "business_areas": business_areas,
                    "applies_to_all_business_areas": not business_areas or business_areas == [""],
                    "subjects": main_doc.get("Subjects", []),
                    "tags": main_doc.get("Tags", []),
                    "document_count": len(documents),
                    "created_at": datetime.utcnow()
                },
                "last_updated": datetime.utcnow()
            }

            logger.debug(f"Summary document created with {len(summary_doc['source_documents'])} sources")
            return summary_doc

        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            raise

    async def _process_document(self, document: Dict, summary_id: ObjectId):
        """Process individual document into chunks"""
        try:
            logger.info(f"Processing document: {document['ComponentPath']}")
            business_areas = document.get("BusinessAreas", [])
            doc_info = {
                "component_path": document["ComponentPath"],
                "page_title": document["PageTitle"],
                "channels": document.get("Channels", []),
                "business_areas": business_areas,
                "applies_to_all_business_areas": not business_areas or business_areas == [""],
                "subjects": document.get("Subjects", []),
                "tags": document.get("Tags", [])
            }

            logger.debug(f"Document info prepared: {doc_info}")

            # Create semantic chunks
            chunks = await self.chunker.create_semantic_chunks(
                document["Content"],
                doc_info
            )

            logger.debug(f"Created {len(chunks)} chunks")
            logger.debug("Chunk types distribution: " + 
                        str({chunk['metadata']['chunk_analysis']['type']: 1 
                            for chunk in chunks}))

            # Add summary_id to each chunk
            for chunk in chunks:
                chunk["summary_id"] = summary_id

            # Store chunks
            if chunks:
                await self.chunks_collection.insert_many(chunks)
                logger.info(f"Stored {len(chunks)} chunks for {document['ComponentPath']}")
                logger.debug(f"Average chunk size: {sum(len(c['content']) for c in chunks)/len(chunks):.0f} characters")
                return chunks
            else:
                logger.warning(f"No chunks created for document {document['ComponentPath']}")
                return []

        except Exception as e:
            logger.error(f"Error processing document {document['ComponentPath']}: {str(e)}")
            raise

    async def _create_summary_embedding(self, summary_text: str, document: Dict) -> List[float]:
        """Create summary embedding with enhanced context"""
        try:
            business_areas = document.get('BusinessAreas', [])
            if business_areas is None:
                business_areas = []
            elif not isinstance(business_areas, (list, tuple)):
                business_areas = [str(business_areas)]
                
            context = f"""
            Document Title: {document.get('PageTitle', '')}
            Business Areas: {', '.join(business_areas)}
            Channels: {', '.join(document.get('Channels', []))}
            
            Summary Content:
            {summary_text}
            """
            
            logger.debug(f"Created context with length: {len(context)}")
            return await self.embeddings.embed_query(context)
            
        except Exception as e:
            logger.error(f"Error creating summary embedding: {str(e)}")
            logger.error(f"BusinessAreas value: {document.get('BusinessAreas')}")
            logger.error(f"BusinessAreas type: {type(document.get('BusinessAreas'))}")
            raise

    def _extract_root_id(self, component_path: str) -> str:
        """Extract root_id from component path"""
        return component_path.split('_')[0]

    async def _store_summary(self, summary_doc: Dict) -> ObjectId:
        """Store summary document"""
        result = await self.summary_collection.insert_one(summary_doc)
        logger.info(f"Stored summary with ID: {result.inserted_id}")
        return result.inserted_id
    
