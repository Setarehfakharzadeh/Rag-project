#!/usr/bin/env python3
"""
Enhanced RAG (Retrieval-Augmented Generation) test script using vector-based retrieval.

This script demonstrates the full vector-based retrieval pipeline:
1. Encoding text with the same model (sentence-transformers)
2. Computing dot product similarity
3. Reducing dimensions
4. Sorting results from best to worst
"""
import json
import sys
import os
import numpy as np
from typing import List, Dict, Any, Tuple
import tempfile
import subprocess
from sklearn.decomposition import PCA

# Check if the mcp_chunks.json file exists
if not os.path.exists('mcp_chunks.json'):
    print("Error: mcp_chunks.json file not found. Please run mcp_fetch.py first.")
    sys.exit(1)

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers package not installed.")
    print("Installing required packages...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "scikit-learn"])
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"Failed to install required packages: {e}")
        print("Please install manually with: pip install sentence-transformers scikit-learn")
        sys.exit(1)

# Load the chunks
with open('mcp_chunks.json', 'r') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} document chunks")

# Global variables for caching
model = None
document_embeddings = []
reduced_embeddings = None
pca_model = None
DIM_REDUCTION_TARGET = 100  # Target dimensions after reduction

def get_model():
    """Load or retrieve the sentence transformer model"""
    global model
    if model is None:
        print("Loading sentence transformer model (this may take a moment)...")
        # We use the all-MiniLM-L6-v2 model which is small but effective
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def encode_text(text: str) -> np.ndarray:
    """
    Step 1: Encode text into a vector embedding
    """
    model = get_model()
    return model.encode(text, show_progress_bar=False)

def compute_document_embeddings() -> List[np.ndarray]:
    """
    Compute and cache embeddings for all document chunks
    """
    global document_embeddings
    if not document_embeddings:
        print("Computing embeddings for all document chunks...")
        model = get_model()
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        # Encode all texts in a batch (more efficient)
        document_embeddings = model.encode(texts, show_progress_bar=True)
    return document_embeddings

def reduce_dimensions(embeddings: List[np.ndarray], force_recompute: bool = False) -> List[np.ndarray]:
    """
    Step 3: Reduce the dimensionality of embeddings
    """
    global reduced_embeddings, pca_model
    
    # If we already computed reduced embeddings and aren't forcing a recompute, return cached
    if reduced_embeddings is not None and not force_recompute:
        return reduced_embeddings
    
    print(f"Reducing embedding dimensions from {embeddings[0].shape[0]} to {DIM_REDUCTION_TARGET}...")
    # Initialize and fit PCA model
    pca_model = PCA(n_components=DIM_REDUCTION_TARGET)
    reduced_embeddings = pca_model.fit_transform(embeddings)
    print("Dimension reduction complete")
    
    return reduced_embeddings

def compute_similarity(query_embedding: np.ndarray, doc_embeddings: List[np.ndarray]) -> List[Tuple[int, float]]:
    """
    Step 2: Compute dot product similarity between query and all documents
    """
    similarity_scores = []
    
    # Normalize query embedding for better similarity comparisons
    query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
    
    for i, doc_embedding in enumerate(doc_embeddings):
        # Normalize document embedding
        doc_embedding_normalized = doc_embedding / np.linalg.norm(doc_embedding)
        # Compute dot product (cosine similarity for normalized vectors)
        similarity = np.dot(query_embedding_normalized, doc_embedding_normalized)
        similarity_scores.append((i, similarity))
    
    return similarity_scores

def rank_results(similarity_scores: List[Tuple[int, float]], top_k: int = 3) -> List[Tuple[int, float]]:
    """
    Step 4: Sort results from best to worst and return top k
    """
    # Sort by similarity score in descending order
    sorted_results = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Return top k with scores > 0
    return [result for result in sorted_results[:top_k] if result[1] > 0]

def vector_search(query: str, top_k: int = 3, use_dim_reduction: bool = True) -> List[Dict[str, Any]]:
    """
    Perform vector-based search with all steps:
    1. Encode query with same model as documents
    2. Compute similarity via dot product
    3. (Optional) Use dimension reduction
    4. Sort results and return top matches
    """
    # Step 1: Encode the query
    query_embedding = encode_text(query)
    
    # Get document embeddings
    all_doc_embeddings = compute_document_embeddings()
    
    # Step 3: Apply dimension reduction if requested
    if use_dim_reduction:
        # Reduce dimensions of document embeddings
        doc_embeddings_reduced = reduce_dimensions(all_doc_embeddings)
        
        # Reduce dimensions of query embedding using the same PCA model
        query_embedding_reduced = pca_model.transform([query_embedding])[0]
        
        # Step 2: Compute similarity with reduced embeddings
        similarity_scores = compute_similarity(query_embedding_reduced, doc_embeddings_reduced)
    else:
        # Step 2: Compute similarity with original embeddings
        similarity_scores = compute_similarity(query_embedding, all_doc_embeddings)
    
    # Step 4: Sort and get top results
    top_results = rank_results(similarity_scores, top_k)
    
    # Return the actual document chunks
    result_chunks = [chunks[idx] for idx, _ in top_results]
    
    # Add similarity scores to results for display
    for i, (idx, score) in enumerate(top_results):
        result_chunks[i]['similarity'] = float(score)
    
    return result_chunks

def generate_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a response based on the query and relevant chunks
    In a real implementation, this would use an LLM API.
    For this demo, we'll just return the chunks.
    """
    if not relevant_chunks:
        return "I couldn't find any relevant information in the MCP documentation for your query."
    
    # In a real implementation, we would prompt an LLM with the context and query
    # For this demo, we'll just show the relevant chunks
    response = f"Here's what I found in the MCP documentation about '{query}':\n\n"
    
    for i, chunk in enumerate(relevant_chunks):
        similarity = chunk.get('similarity', 0.0)
        response += f"--- From {chunk['source']} (Similarity: {similarity:.4f}) ---\n"
        response += chunk['content'][:300] + "...\n\n"  # Only show first 300 chars
    
    response += "\nNote: This is a vector-based RAG demo without an actual LLM backend."
    return response

def main():
    """Main function for the enhanced RAG demo"""
    print("MCP Documentation Vector-Based RAG Demo")
    print("Enter 'quit' to exit")
    print("-" * 60)
    
    # Pre-compute embeddings and dimension reduction
    compute_document_embeddings()
    reduce_dimensions(document_embeddings)
    
    print("\nSetup complete! You can now ask questions.")
    
    while True:
        query = input("\nEnter your question about MCP: ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query.strip():
            continue
            
        # Retrieve relevant chunks using vector search
        print("Searching with vector embeddings...")
        relevant_chunks = vector_search(query, top_k=3, use_dim_reduction=True)
        
        # Generate and print the response
        response = generate_response(query, relevant_chunks)
        print("\n" + response)

if __name__ == "__main__":
    main() 