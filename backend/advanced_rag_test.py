#!/usr/bin/env python3
"""
Advanced RAG (Retrieval-Augmented Generation) test script using MCP documentation
and the Llama model.
"""
import json
import sys
import os
from typing import List, Dict, Any
import subprocess
import tempfile

# Check if the mcp_chunks.json file exists
if not os.path.exists('mcp_chunks.json'):
    print("Error: mcp_chunks.json file not found. Please run mcp_fetch.py first.")
    sys.exit(1)

# Check if the model file exists
MODEL_PATH = "../stablebeluga-7b.Q4_K_M.gguf"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure the model file is available.")
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

def generate_response_with_llama(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a response using the Llama model with the relevant chunks as context
    """
    if not relevant_chunks:
        return "I couldn't find any relevant information in the MCP documentation for your query."
    
    # Prepare the context from the relevant chunks
    context = "Here's some information from the MCP documentation:\n\n"
    for chunk in relevant_chunks:
        context += f"--- From {chunk['source']} ---\n"
        context += chunk['content'][:800] + "\n\n"  # Limit chunk size
    
    # Create the prompt for the LLM
    prompt = f"""
{context}

Based on the information above, please answer the following question:
{query}

Keep your answer concise and focused on information from the provided documentation. 
If the answer is not in the documentation, please say so.
"""
    
    # Create a temporary file for the prompt
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        prompt_file = f.name
        f.write(prompt)
    
    try:
        # Run the Llama model with the prompt
        print("Generating response with the Llama model...")
        
        # Run the llama.cpp CLI with our model and prompt
        # Adjust parameters as needed
        cmd = [
            "llama-cli", "generate",
            "-m", MODEL_PATH,
            "-f", prompt_file,
            "--temp", "0.1",
            "--top-p", "0.9",
            "--max-tokens", "500",
            "--stop", "\n\n\n"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            response = result.stdout.strip()
            
            # Extract just the generated response (after the prompt)
            if response:
                # Find where the prompt ends and the generation begins
                response_start = response.find(query) + len(query)
                if response_start > 0 and response_start < len(response):
                    response = response[response_start:].strip()
            
            if not response:
                response = "Sorry, I couldn't generate a response with the model."
                
        except subprocess.CalledProcessError as e:
            print(f"Error running Llama model: {e}")
            print(f"Error output: {e.stderr}")
            response = "Error: Failed to generate response with the Llama model."
        except FileNotFoundError:
            print("Error: llama-cli command not found. Using fallback response.")
            # Fallback to a simple response without the model
            response = fallback_response(query, relevant_chunks)
    finally:
        # Clean up the temporary file
        os.unlink(prompt_file)
    
    return response

def fallback_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a fallback response without using an LLM
    """
    response = f"Here's what I found in the MCP documentation about '{query}':\n\n"
    
    for i, chunk in enumerate(relevant_chunks):
        response += f"--- From {chunk['source']} ---\n"
        response += chunk['content'][:300] + "...\n\n"
    
    response += "\nNote: This is using a fallback method as the Llama model is not available."
    return response

def main():
    """Main function for the advanced RAG demo"""
    print("Advanced MCP Documentation RAG Demo with Llama Model")
    print("Enter 'quit' to exit")
    print("-" * 60)
    
    while True:
        query = input("\nEnter your question about MCP: ")
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query.strip():
            continue
            
        # Retrieve relevant chunks
        relevant_chunks = simple_search(query, chunks, top_k=3)
        
        # Generate and print the response
        try:
            response = generate_response_with_llama(query, relevant_chunks)
        except Exception as e:
            print(f"Error generating response: {e}")
            response = fallback_response(query, relevant_chunks)
        
        print("\n" + response)

if __name__ == "__main__":
    main() 