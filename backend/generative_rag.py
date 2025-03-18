#!/usr/bin/env python3
"""
Generative RAG implementation using vector-based retrieval with OpenAI's ChatGPT API.

This script combines:
1. Vector-based retrieval from MCP documentation
2. Generation using OpenAI's GPT model
"""
import json
import sys
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import requests
from dotenv import load_dotenv
from sklearn.decomposition import PCA

# Load environment variables from .env file if it exists
load_dotenv()

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
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "sentence-transformers", "scikit-learn", 
                              "python-dotenv", "openai"])
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"Failed to install required packages: {e}")
        print("Please install manually with: pip install sentence-transformers scikit-learn python-dotenv openai")
        sys.exit(1)

# Try to import OpenAI, but don't fail if not available (we'll provide a fallback)
try:
    import openai
    OPENAI_AVAILABLE = True
    # Get API key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI package not available. Will use fallback response generation.")

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
    Encode text into a vector embedding
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
    Reduce the dimensionality of embeddings
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
    Compute dot product similarity between query and all documents
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
    Sort results from best to worst and return top k
    """
    # Sort by similarity score in descending order
    sorted_results = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Return top k with scores > 0
    return [result for result in sorted_results[:top_k] if result[1] > 0]

def vector_search(query: str, top_k: int = 5, use_dim_reduction: bool = True) -> List[Dict[str, Any]]:
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

def generate_openai_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a response using OpenAI's GPT models with the relevant chunks as context
    """
    if not OPENAI_AVAILABLE or not openai.api_key:
        return None
    
    try:
        # Prepare the context from the relevant chunks
        context = "Here's some information from the MCP documentation:\n\n"
        for chunk in relevant_chunks:
            context += f"--- From {chunk['source']} ---\n"
            context += chunk['content'][:800] + "\n\n"
        
        # Create messages for ChatGPT
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps users understand the Model Context Protocol (MCP). "
                                         "Respond based on the provided documentation context. If the information is not in the context, "
                                         "say that you don't know or cannot find the information in the documentation."},
            {"role": "user", "content": f"{context}\n\nBased on the information above, please answer this question: {query}"}
        ]
        
        # Call the OpenAI API
        print("Generating response with OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if available
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            n=1
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return None

def generate_gemini_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Generate a response using Google's Gemini API with the relevant chunks as context
    """
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        # Prepare the context from the relevant chunks
        context = "Here's some information from the MCP documentation:\n\n"
        for chunk in relevant_chunks:
            context += f"--- From {chunk['source']} ---\n"
            context += chunk['content'][:800] + "\n\n"
        
        # Create the prompt for Gemini
        prompt = f"{context}\n\nBased on the information above, please answer this question: {query}\n\nRespond based on the provided documentation context. If the information is not in the context, say that you don't know or cannot find the information in the documentation."
        
        # Make the API request to Gemini
        print("Generating response with Google Gemini...")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500
            }
        }
        
        response = requests.post(
            f"{url}?key={api_key}",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if 'candidates' in response_json and len(response_json['candidates']) > 0:
                return response_json['candidates'][0]['content']['parts'][0]['text']
        
        print(f"Failed to get response from Gemini API: {response.status_code}")
        print(response.text)
        return None
    except Exception as e:
        print(f"Error generating response with Gemini: {e}")
        return None

def generate_response(query: str, relevant_chunks: List[Dict[str, Any]], use_gemini: bool = False) -> str:
    """
    Generate a response based on the query and relevant chunks
    Try to use a generative model if available, otherwise fall back to a simpler response
    """
    if not relevant_chunks:
        return "I couldn't find any relevant information in the MCP documentation for your query."
    
    # Try Gemini if requested
    if use_gemini:
        gemini_response = generate_gemini_response(query, relevant_chunks)
        if gemini_response:
            return gemini_response
    
    # Try OpenAI if available
    if OPENAI_AVAILABLE and openai.api_key:
        openai_response = generate_openai_response(query, relevant_chunks)
        if openai_response:
            return openai_response
    
    # Fallback response if no generative model is available
    print("No generative model available, using fallback response.")
    response = f"Here's what I found in the MCP documentation about '{query}':\n\n"
    
    for i, chunk in enumerate(relevant_chunks):
        similarity = chunk.get('similarity', 0.0)
        response += f"--- From {chunk['source']} (Similarity: {similarity:.4f}) ---\n"
        response += chunk['content'][:300] + "...\n\n"
    
    response += "\nNote: This is a fallback response as no generative model (OpenAI/Gemini) is available."
    return response

def select_model() -> bool:
    """Ask the user which model they want to use"""
    print("\nPlease select which generative model you'd like to use:")
    print("1. OpenAI GPT (requires API key in OPENAI_API_KEY environment variable)")
    print("2. Google Gemini (requires API key in GEMINI_API_KEY environment variable)")
    print("3. No generative model (fallback to retrieval only)")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set.")
            key = input("Enter your OpenAI API key (or press Enter to skip): ")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                openai.api_key = key
            else:
                print("No API key provided, will use fallback response.")
        return False  # Not using Gemini
        
    elif choice == '2':
        if not os.getenv("GEMINI_API_KEY"):
            print("Warning: GEMINI_API_KEY environment variable not set.")
            key = input("Enter your Gemini API key (or press Enter to skip): ")
            if key:
                os.environ["GEMINI_API_KEY"] = key
            else:
                print("No API key provided, will use fallback response.")
        return True  # Using Gemini
        
    else:
        print("Using retrieval only (no generative model).")
        return False

def main():
    """Main function for the generative RAG demo"""
    print("MCP Documentation Generative RAG Demo")
    print("=" * 60)
    print("This demo combines vector-based retrieval with generative AI.")
    print("=" * 60)
    
    # Pre-compute embeddings and dimension reduction
    compute_document_embeddings()
    reduce_dimensions(document_embeddings)
    
    # Ask user which model to use
    use_gemini = select_model()
    
    print("\nSetup complete! You can now ask questions.")
    print("Enter 'quit' to exit")
    print("-" * 60)
    
    while True:
        query = input("\nEnter your question about MCP: ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query.strip():
            continue
            
        # Retrieve relevant chunks using vector search
        print("Searching with vector embeddings...")
        start_time = time.time()
        relevant_chunks = vector_search(query, top_k=5, use_dim_reduction=True)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks in {time.time() - start_time:.2f} seconds")
        
        # Generate and print the response
        try:
            start_time = time.time()
            response = generate_response(query, relevant_chunks, use_gemini=use_gemini)
            print(f"Generated response in {time.time() - start_time:.2f} seconds")
            print("\n" + response)
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main() 