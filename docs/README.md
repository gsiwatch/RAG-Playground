# RAG-Optimized Document Processor for Hierarchical Content

## Background and Problem
The challenge was to process hierarchically structured policy documents where:
1. Documents are interconnected through ComponentPath (e.g., x37806_x111544_x111562)
2. Content has natural hierarchies (definitions → general info → procedures)
3. Documents contain product-specific variations and requirements
4. Information relationships and product context must be preserved for accurate retrieval

## Solution Architecture

### 1. Document Processing Layer
- **Component Path Analysis**
  - Root level (x37806): Main topic/policy
  - Second level (x111544): Product category
  - Third level (x111562): Content type
  - Automated product context detection
  - Relationship mapping between components

- **Content Organization**
  - Definitions and core concepts
  - Product-specific requirements
  - Procedures and conditions
  - Cross-product variations

### 2. Smart Processing Layer
a) **Enhanced Content Analysis**
   - Document structure analysis
   - Product relationship mapping
   - Content type classification
   - Semantic boundary detection

b) **Hierarchical Summaries**
   - Main policy summary
   - Section-specific summaries
   - Product-specific context
   - Relationship-aware summaries

c) **Context-Aware Chunking**
   - Dynamic chunk sizing
   - Semantic boundary preservation
   - Product context maintenance
   - Relationship preservation

### 3. Storage Strategy
- **Summary Collection**:
  ```python
  {
      'root_id': 'x37806',
      'page_title': 'PACE Lien',
      'main_summary': {
          'text': '...',
          'embedding': [...]
      },
      'section_summaries': {
          'definition': {...},
          'procedure': {...}
      },
      'content_hierarchy': {...},
      'products': ['VA', 'FHA', ...]
  }
   ```
- **Chunks Collection:**
```python
{
    'content': '...',
    'embedding': [...],
    'metadata': {
        'component_path': 'x37806_x111544_x111562',
        'products': ['VA', 'FHA'],
        'content_type': 'procedure',
        'section_type': 'requirement'
    },
    'context': {
        'section_summary': '...',
        'product_context': {...}
    }
}
```

# Enhanced Chunking Strategy
1. Content-Aware Parameters:
   - Dynamic sizing based on content type
   - Product-specific boundary detection
   - Semantic relationship preservation
   - Token optimization for embeddings
2. Configuration:
```
CHUNKING_CONFIG = {
    'content': {
        'overlap': 50,     # Context continuity
        'min_size': 100,   # Minimum meaningful unit
        'max_size': 1500,  # Context balance
        'max_tokens': 1500 # Embedding limit
    },
    'summary': {
        'min_size': 100,
        'max_size': 1000,
        'max_tokens': 1000
    }
}
```
# Processing Flow

1. Enhanced Document Retrieval:
   - Component path analysis
   - Product context extraction
   - Relationship mapping
   - Content type classification
2. Smart Content Processing:
   - Semantic chunking
   - Product context preservation
   - Relationship detection
   - Context embedding
3. Storage and Indexing:
   - Vector indexes for embeddings
   - Product-specific indexes
   - Relationship preservation
   - Context maintenance
   - Query Processing

1. Query Classification:
```
query_types = [
    'general',           # Overview queries
    'product_specific',  # Product requirements
    'comparison',        # Cross-product comparison
    'procedure'          # Specific procedures
]
```
2. Context-Aware Search:
   - Product context detection
   - Relationship-aware retrieval
   - Semantic similarity matching
   - Context preservation
3. Response Generation:
   - Context-aware formatting
   - Product-specific details
   - Relationship inclusion
   - Source citation

# Example Flows
```
1. Product-Specific Query
Query: "What are the VA loan modification requirements for PACE liens?"
# Processing Flow:
1. Query classification: product_specific
2. Product detection: VA
3. Content type: requirement
4. Context retrieval: loan_modification
5. Response generation with citations
2. Comparison Query

Query: "Compare VA and FHA PACE lien requirements"
# Processing Flow:
1. Query classification: comparison
2. Products: [VA, FHA]
3. Requirement comparison
4. Difference highlighting
5. Structured response
```
# Testing and Validation: 
1. Content Processing:
   - Chunk integrity
   - Product context accuracy
   - Relationship preservation
   - Embedding quality
   - Query Processing:
2. Response accuracy: 
   - Context preservation
   - Product specificity
   - Relationship awareness
   - Future Enhancements
3. Processing Improvements:
   - Real-time content updates
   - Dynamic relationship mapping
   - Automated product detection
   - Enhanced semantic analysis
4. Query Handling:
   - Multi-product comparisons
   - Complex relationship queries
   - Temporal context awareness
   - User feedback incorporation

# Usage
```
# Initialize processor
processor = await DocumentProcessor.create()

# Process documents
processor.process_page_documents("x37806")

# Query processing
rag_system = RAGSystem()
response = await rag_system.process_query(
    query="What are VA PACE requirements?",
    channel="Servicing"
)
```

## Implementation Details

### Collections and Indexes
1. Summary Collection Index:
```json
{
    "mappings": {
        "dynamic": false,
        "fields": {
            "combined_embedding": {
                "dimensions": 1536,
                "similarity": "cosine",
                "type": "knnVector"
            },
            "products": {
                "type": "string"
            }
        }
    }
}
```
2. Chunks Collection Index:
```json
{
    "mappings": {
        "dynamic": false,
        "fields": {
            "embedding": {
                "dimensions": 1536,
                "similarity": "cosine",
                "type": "knnVector"
            },
            "metadata.products": {
                "type": "string"
            },
            "metadata.section_type": {
                "type": "string"
            }
        }
    }
}
```

# Search Strategy
1. Summary Search (General Queries):
```python
search_pipeline = [
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "combined_embedding",
            "numCandidates": 50,
            "limit": 1,
            "index": "complied_answers_testing_summary_embeddings"
        }
    }
]
```
2. Detailed Search (Specific Queries):
```python
search_pipeline = [
    {
        "$vectorSearch": {
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 50,
            "limit": 10,
            "index": "complied_answers_test_chunks_embeddings"
        }
    }
]
```


## Implementation Notes

### Why Two Collections?
1. Summary Collection:
   - Efficient handling of general queries
   - Maintains hierarchical context
   - Reduces number of vector searches

2. Chunks Collection:
   - Detailed information retrieval
   - Product-specific queries
   - Precise answer generation

### Semantic Chunking Benefits
1. Context Preservation:
   - Maintains semantic relationships
   - Preserves product-specific details
   - Keeps requirements together

2. Better Retrieval:
   - More accurate vector search
   - Meaningful chunk boundaries
   - Reduced noise in results

### Query Classification Purpose
1. Optimizes search strategy
2. Determines response format
3. Guides confidence scoring
