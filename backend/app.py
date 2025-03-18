from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

# Load the model
MODEL_PATH = "../stablebeluga-7b.Q4_K_M.gguf"
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1
)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load MCP documentation chunks
MCP_CHUNKS_PATH = "mcp_chunks.json"

# If the chunks file doesn't exist yet, create a placeholder
if not os.path.exists(MCP_CHUNKS_PATH):
    with open(MCP_CHUNKS_PATH, "w") as f:
        json.dump([], f)

with open(MCP_CHUNKS_PATH, "r") as f:
    mcp_chunks = json.load(f)

# Create embeddings for chunks (if not already created)
EMBEDDINGS_PATH = "mcp_embeddings.npy"
if os.path.exists(EMBEDDINGS_PATH) and len(mcp_chunks) > 0:
    chunk_embeddings = np.load(EMBEDDINGS_PATH)
else:
    # Create embeddings if chunks exist
    if len(mcp_chunks) > 0:
        texts = [chunk["content"] for chunk in mcp_chunks]
        chunk_embeddings = embedding_model.encode(texts)
        np.save(EMBEDDINGS_PATH, chunk_embeddings)
    else:
        chunk_embeddings = np.array([])

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve the most relevant chunks for a query"""
    if len(mcp_chunks) == 0:
        return []
    
    # Get query embedding
    query_embedding = embedding_model.encode(query)
    
    # Calculate similarity
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    # Get top k chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [mcp_chunks[i] for i in top_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Retrieve relevant documents for RAG
        relevant_chunks = retrieve_relevant_chunks(user_message)
        
        # Format context from relevant chunks
        context = ""
        if relevant_chunks:
            context = "Here is some relevant information about Model Context Protocol:\n\n"
            for chunk in relevant_chunks:
                context += f"{chunk['content']}\n\n"
                context += f"(Source: {chunk['source']})\n\n"
        
        # Create prompt with retrieved context
        if context:
            prompt = f"""You are a helpful assistant specializing in the Model Context Protocol.
            
{context}

Based on the information above, please respond to the following question:
{user_message}

If the information provided doesn't contain the answer, please say so and provide general information about MCP if possible."""
        else:
            prompt = f"""You are a helpful assistant specializing in the Model Context Protocol.

Please respond to the following question about MCP (Model Context Protocol):
{user_message}

If you don't have specific information, please let the user know and suggest they check the official documentation at modelcontextprotocol.io."""
        
        # Generate response
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["Human:", "User:"],
            echo=False
        )
        
        return jsonify({
            'success': True,
            'response': response['choices'][0]['text'].strip()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
