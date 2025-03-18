#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) test script using MCP documentation.
"""
import json
import sys
import os
from typing import List, Dict, Any

# Check if the mcp_chunks.json file exists
if not os.path.exists('mcp_chunks.json'):
    print("Error: mcp_chunks.json file not found. Please run mcp_fetch.py first.")
    sys.exit(1)

# Load the chunks
with open('mcp_chunks.json', 'r') as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} document chunks")

def simple_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Simple keyword-based search (this would be replaced by a vector database in a real implementation)
    """
    # Convert query to lowercase for case-insensitive matching
    query = query.lower()
    # Split the query into keywords
    keywords = query.split()
    
    # Calculate a simple score based on keyword matches
    chunk_scores = []
    for i, chunk in enumerate(chunks):
        content = chunk['content'].lower()
        score = sum(1 for keyword in keywords if keyword in content)
        chunk_scores.append((i, score))
    
    # Sort by score (descending) and return top_k
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunks[idx] for idx, score in chunk_scores[:top_k] if score > 0]
    
    return top_chunks

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
        response += f"--- From {chunk['source']} ---\n"
        response += chunk['content'][:300] + "...\n\n"  # Only show first 300 chars
    
    response += "\nNote: This is a simple RAG demo without an actual LLM backend."
    return response

def main():
    """Main function for the RAG demo"""
    print("MCP Documentation RAG Demo")
    print("Enter 'quit' to exit")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your question about MCP: ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query.strip():
            continue
            
        # Retrieve relevant chunks
        relevant_chunks = simple_search(query, chunks)
        
        # Generate and print the response
        response = generate_response(query, relevant_chunks)
        print("\n" + response)

if __name__ == "__main__":
    main() 