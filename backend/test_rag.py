"""
Test script for the RAG system with MCP documentation.
This allows testing retrieval without running the full Flask server.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load MCP documentation chunks
try:
    with open("mcp_chunks.json", "r") as f:
        mcp_chunks = json.load(f)
    print(f"Loaded {len(mcp_chunks)} chunks from mcp_chunks.json")
except FileNotFoundError:
    print("Warning: mcp_chunks.json not found. Run crawl_mcp.py first.")
    mcp_chunks = []

# Load embeddings
try:
    chunk_embeddings = np.load("mcp_embeddings.npy")
    print(f"Loaded embeddings with shape {chunk_embeddings.shape}")
except FileNotFoundError:
    print("Warning: mcp_embeddings.npy not found. Run crawl_mcp.py first.")
    chunk_embeddings = np.array([])

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve the most relevant chunks for a query"""
    if len(mcp_chunks) == 0 or len(chunk_embeddings) == 0:
        print("No chunks or embeddings available")
        return []
    
    # Get query embedding
    query_embedding = embedding_model.encode(query)
    
    # Calculate similarity
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(mcp_chunks[i], similarities[i]) for i in top_indices]

def test_retrieval():
    """Test retrieval with sample queries"""
    test_queries = [
        "How do I set up a Model Context Protocol server?",
        "What is the Model Context Protocol?",
        "How can I implement a client for MCP?",
        "What are the main components of the protocol?",
        "How do I handle errors in MCP?"
    ]
    
    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("=" * 50)
        
        relevant_chunks = retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            print("No relevant chunks found.")
            continue
            
        for i, (chunk, similarity) in enumerate(relevant_chunks):
            print(f"\nResult {i+1} (Similarity: {similarity:.4f})")
            print("-" * 50)
            print(f"Source: {chunk['source']}")
            print(f"Content: {chunk['content'][:300]}...")

if __name__ == "__main__":
    if len(mcp_chunks) > 0 and len(chunk_embeddings) > 0:
        test_retrieval()
    else:
        print("Cannot test retrieval without chunks and embeddings.")
        print("Please run crawl_mcp.py first to generate the required data.") 