# NOTE: please run this ONLY when you are aware about the impact it will be making
# Steps to use this script
# python3 -m scripts.run_ingestion <root_id>
# or 
# python3 -m scripts.run_ingestion (now it will run on the entire document)

import asyncio
from typing import Optional, List
from tqdm import tqdm
from rag_strategies.ingestion.processor import DocumentProcessor
from rag_strategies.config import settings
from rag_strategies.utils.logger import setup_logger
import math

logger = setup_logger(__name__)

async def get_unique_root_ids(processor: DocumentProcessor) -> List[str]:
    """Get all unique root IDs from ComponentPath field"""
    try:
        pipeline = [
            {
                "$project": {
                    "root_id": {"$substr": ["$ComponentPath", 0, {"$indexOfCP": ["$ComponentPath", "_"]}]}
                }
            },
            {"$group": {"_id": "$root_id"}},
            {"$sort": {"_id": 1}}
        ]
        
        results = await processor.docs_collection.aggregate(pipeline).to_list(None)
        root_ids = [doc["_id"] for doc in results]
        logger.info(f"Found {len(root_ids)} unique root IDs")
        logger.info(f"Sample root IDs: {root_ids[:5] if len(root_ids) > 5 else root_ids}")
        return root_ids

    except Exception as e:
        logger.error(f"Error getting unique root IDs: {str(e)}")
        raise

async def verify_collections_and_indexes(processor):
    """Verify collections exist and create required indexes"""
    try:
        collections = await processor.db.list_collection_names()
        
        if settings.mongodb_collection_name not in collections:
            raise ValueError(f"Source collection {settings.mongodb_collection_name} does not exist")
        
        # Create destination collections if they don't exist
        for collection_name in [settings.mongodb_chunks_collection, settings.mongodb_summary_collection]:
            if collection_name not in collections:
                logger.info(f"Creating collection: {collection_name}")
                await processor.db.create_collection(collection_name)
        
        logger.info("Setting up indexes...")
        await processor.chunks_collection.create_index([("summary_id", 1)])
        await processor.summary_collection.create_index([("root_id", 1)], unique=True)
        
        logger.info("Collections and indexes verified")
        
    except Exception as e:
        logger.error(f"Error in collection/index verification: {str(e)}")
        raise

async def process_batch(processor: DocumentProcessor, root_ids_batch: List[str], worker_id: int) -> tuple:
    """Process a batch of root IDs"""
    successful = 0
    failed = 0
    
    logger.info(f"Worker {worker_id} starting to process {len(root_ids_batch)} root IDs")
    logger.info(f"Worker {worker_id} batch range: {root_ids_batch[0]} to {root_ids_batch[-1]}")
    
    for root_id in root_ids_batch:
        try:
            # Delete existing data for this root_id before processing
            logger.info(f"Worker {worker_id} - Cleaning up existing data for root_id: {root_id}")
            await processor.chunks_collection.delete_many({"root_id": root_id})
            await processor.summary_collection.delete_many({"root_id": root_id})
            
            await processor.process_root_documents(root_id)
            successful += 1
            logger.info(f"Worker {worker_id} - Successfully processed root_id: {root_id}")
        except Exception as e:
            logger.error(f"Worker {worker_id} - Error processing root_id {root_id}: {str(e)}")
            failed += 1
    
    logger.info(f"Worker {worker_id} completed: {successful} successful, {failed} failed")
    return successful, failed

async def run_ingestion(root_id: Optional[str] = None):
    """Run ingestion for a specific root_id or all documents"""
    ROOT_IDS = []

    logger.info("Starting ingestion process...")
    processor = await DocumentProcessor.create()
    
    try:
        await verify_collections_and_indexes(processor)
        
        if root_id:
            logger.info(f"Cleaning up existing data for root_id: {root_id}")
            await processor.chunks_collection.delete_many({"root_id": root_id})
            await processor.summary_collection.delete_many({"root_id": root_id})
            
            # Process specific root_id
            logger.info(f"Processing single root_id: {root_id}")
            await processor.process_root_documents(root_id)
            logger.info(f"Completed processing root_id: {root_id}")
        else:
            logger.info("Starting batch processing for all root IDs")
            
            # Use predefined ROOT_IDS instead of querying the database
            root_ids = ROOT_IDS
            total_roots = len(root_ids)
            
            num_workers = 10
            batch_size = math.ceil(total_roots / num_workers)
            
            batches = []
            for i in range(0, total_roots, batch_size):
                batch = root_ids[i:min(i + batch_size, total_roots)]
                batches.append(batch)
            
            logger.info(f"Created {len(batches)} batches:")
            for i, batch in enumerate(batches):
                logger.info(f"Batch {i+1}: Size={len(batch)}, First={batch[0]}, Last={batch[-1]}")
            
            processors = [await DocumentProcessor.create() for _ in range(len(batches))]
            logger.info(f"Created {len(processors)} processor instances")
            
            tasks = []
            for i, batch in enumerate(batches):
                task = asyncio.create_task(process_batch(processors[i], batch, i+1))
                tasks.append(task)
            
            successful = 0
            failed = 0
            batch_results = {}
            
            with tqdm(total=total_roots, desc="Processing root IDs") as pbar:
                for i, completed_task in enumerate(asyncio.as_completed(tasks)):
                    batch_successful, batch_failed = await completed_task
                    successful += batch_successful
                    failed += batch_failed
                    batch_results[i] = (batch_successful, batch_failed)
                    pbar.update(batch_successful + batch_failed)
            
            logger.info("\nBatch Processing Results:")
            for batch_num, (batch_successful, batch_failed) in batch_results.items():
                logger.info(f"Batch {batch_num + 1}: Successful={batch_successful}, Failed={batch_failed}, "
                          f"Total={batch_successful + batch_failed}")
            
            for proc in processors:
                await proc.cleanup()
            
            logger.info("=" * 50)
            logger.info("Final Ingestion Summary:")
            logger.info(f"Total Root IDs: {total_roots}")
            logger.info(f"Total Successful: {successful}")
            logger.info(f"Total Failed: {failed}")
            logger.info(f"Total Processed: {successful + failed}")
            logger.info(f"Number of Batches: {len(batches)}")
            logger.info(f"Batch Size: {batch_size}")
            logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise
    finally:
        if processor:
            await processor.cleanup()
            logger.info("Cleanup completed")

if __name__ == "__main__":
    import sys
    
    # Get optional root_id from command line
    root_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        if root_id and not root_id.startswith('x'):
            raise ValueError("Root ID must start with 'x'")
            
        asyncio.run(run_ingestion(root_id))
        logger.info("Script completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)
