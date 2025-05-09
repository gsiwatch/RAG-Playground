import asyncio
from datetime import datetime
from typing import Optional, Dict

from rag_strategies.retrieval.rag_system import RAGSystem
from rag_strategies.utils.logger import setup_logger

logger = setup_logger(__name__)

async def display_response(response: Dict, query: str):
    """Display response in a clean format"""
    print("\n" + "=" * 80)
    print("\nQ: " + query)
    print("-" * 80)
    print("\nA: " + response.get("answer", "No answer available"))
    print("\n" + "=" * 80)
    
    print("\nConfidence Score:", f"{response.get('confidence', 0):.2f}")
    
    if response.get("citations"):
        print("\nSources:")
        print("-" * 40)
        for idx, citation in enumerate(response.get("citations", []), 1):
            print(f"\n[{idx}] Content Preview: {citation['content'][:150]}...")
            if citation.get('metadata'):
                print(f"Document Path: {citation['metadata'].get('document_path', 'N/A')}")

async def test_retrieval(
    query: str,
    metadata: Optional[Dict] = None,
    verbose: bool = False
) -> Dict:
    """Test retrieval system"""
    start_time = datetime.now()
    
    if verbose:
        logger.info(f"Processing query: {query}")
    
    try:
        # Initialize RAG system
        rag_system = await RAGSystem.create()
        
        # Process query
        response = await rag_system.process_query(
            query=query,
            metadata=metadata
        )
        
        # Display response
        await display_response(response, query)
        
        if verbose:
            # Log processing details
            logger.info("\nProcessing Details:")
            logger.info("-" * 40)
            logger.info(f"Sources used: {response['metadata'].get('sources_used', {})}")
            logger.info(f"Processing time: {response['metadata'].get('processed_at')}")
            
        await rag_system.cleanup()
        return response

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

async def run_test_cases():
    """Run multiple test cases"""
    test_cases = [
        {
            "query": "Does a repair need to be completed prior to close?",
            "description": "Repair"
        }
    ]
    
    results = []
    for test_case in test_cases:
        print("\n" + "=" * 80)
        print(f"Test Case: {test_case['description']}")
        print("=" * 80)
        
        try:
            response = await test_retrieval(
                query=test_case['query'],
                metadata={"test_case": test_case['description']},
                verbose=False
            )
            
            results.append({
                "description": test_case['description'],
                "success": True,
                "confidence": response.get('confidence', 0),
                "has_citations": bool(response.get('citations'))
            })
            
        except Exception as e:
            logger.error(f"Test case failed: {str(e)}")
            results.append({
                "description": test_case['description'],
                "success": False,
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    async def main():
        # Run single test
        # TEST_QUERY = "Can the property be considered a primary residence?"
        # TEST_QUERY = "Does a repair need to be completed prior to close?"
        # TEST_QUERY = "Is an appraisal addendum required?"
        # TEST_QUERY = "Do payroll deposits need to be sourced?"
        # TEST_QUERY = "Do payroll deposits need to be sourced?"
        # TEST_QUERY = "Does an American Express payment need to be included in DTI on FHA?"
        # TEST_QUERY = "Does an American Express payment need to be included in DTI on FHA?"
        # TEST_QUERY = "Does Self Employment loss need to be included in DTI?"
        TEST_QUERY = "Does Self Employment loss need to be included in DTI for Freddie? are there any exclusions"
        await test_retrieval(
            query=TEST_QUERY,
            metadata={"test_run": True},
            verbose=True
        )
        
        # Uncomment to run all test cases
        # results = await run_test_cases()
        # print("\nTest Results:", results)
    
    asyncio.run(main())
