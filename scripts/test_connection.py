import asyncio
from pymongo import MongoClient
from langchain_openai import ChatOpenAI 

from rag_strategies.config import settings
from rag_strategies.utils import setup_logger, setup_ssl_certificates

logger = setup_logger(__name__)
setup_ssl_certificates()

async def test_openai_connection():
    logger.info("Testing OpenAI connection...")
    try:
        chat = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model_name,
        )
        response = await chat.ainvoke(
            "Say 'Hello'"
        )
        logger.info(f"OpenAI connection successful!: {response}")
    except Exception as e:
        logger.error(f"OpenAI connection failed: {str(e)}")
        raise

def test_mongodb_connection():
    logger.info(f"Testing MongoDB connection...")
    try:
        client = MongoClient(settings.mongodb_uri)
        
        # First, let's see what databases are available
        databases = client.list_database_names()
        logger.info(f"Available databases: {databases}")

        # Check collections in the assigned database
        db = client[settings.mongodb_db_name]
        collections = db.list_collection_names()
        logger.info(f"Collections in 'answer' database: {collections}")

        # Specifically check for 'compiled-answers-testing' collection
        if 'compiled-answers-testing' in collections:
            logger.info("Found 'compiled-answers-testing' collection!")
            count = db['compiled-answers-testing'].count_documents({})
            logger.info(f"Number of documents in compiled-answers-testing: {count}")
        else:
            logger.info("'compiled-answers-testing' collection not found")

    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise
    finally:
        client.close()
    
async def test_connections():
    try:
        # Test OpenAI first
        await test_openai_connection()
        logger.info("OpenAI test completed, proceeding to MongoDB test...")
        
        # Then test MongoDB
        test_mongodb_connection()
        logger.info("All connection tests completed successfully!")
    except Exception as e:
        logger.error(f"Connection testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_connections())
