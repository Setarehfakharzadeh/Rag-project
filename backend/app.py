from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from llama_cpp import Llama
from dotenv import load_dotenv
import sys

# Try to import the generative_rag module for OpenAI and Gemini integration
try:
    from generative_rag import generate_openai_response, generate_gemini_response
    GENERATIVE_RAG_AVAILABLE = True
except ImportError:
    GENERATIVE_RAG_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load the model
MODEL_PATH = "../stablebeluga-7b.Q4_K_M.gguf"

# Only load the model if not in testing mode
if not app.config.get('TESTING', False):
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_gpu_layers=-1
        )
    except Exception as e:
        print(f"Warning: Could not load language model: {e}")
        llm = None
else:
    llm = None

# Load the embedding model
if not app.config.get('TESTING', False):
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Warning: Could not load embedding model: {e}")
        embedding_model = None
else:
    embedding_model = None

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
if os.path.exists(EMBEDDINGS_PATH) and len(mcp_chunks) > 0 and not app.config.get('TESTING', False):
    chunk_embeddings = np.load(EMBEDDINGS_PATH)
else:
    # Create embeddings if chunks exist and not in testing mode
    if len(mcp_chunks) > 0 and embedding_model is not None and not app.config.get('TESTING', False):
        texts = [chunk["content"] for chunk in mcp_chunks]
        chunk_embeddings = embedding_model.encode(texts)
        np.save(EMBEDDINGS_PATH, chunk_embeddings)
    else:
        chunk_embeddings = np.array([])

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve the most relevant chunks for a query"""
    if len(mcp_chunks) == 0:
        return []
    
    # Check if embedding model is available
    if embedding_model is None:
        return []
    
    # Get query embedding
    query_embedding = embedding_model.encode(query)
    
    # Check if chunk embeddings exist
    if len(chunk_embeddings) == 0:
        return []
    
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
        model_choice = data.get('model', 'local')  # Default to local model
        
        # Check if we're in testing mode
        if app.config.get('TESTING', False):
            return jsonify({
                'success': True,
                'response': 'This is a test response. The actual model is not loaded in testing mode.'
            })
        
        # Retrieve relevant documents for RAG
        try:
            relevant_chunks = retrieve_relevant_chunks(user_message)
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            relevant_chunks = []
        
        # Format context from relevant chunks
        context = ""
        if relevant_chunks:
            context = "Here is some relevant information about Model Context Protocol:\n\n"
            for chunk in relevant_chunks:
                context += f"{chunk['content']}\n\n"
                context += f"(Source: {chunk['source']})\n\n"
        
        # Handle different model choices
        if model_choice == 'openai' and GENERATIVE_RAG_AVAILABLE:
            # Use OpenAI API
            try:
                model_response = generate_openai_response(user_message, relevant_chunks)
                if not model_response:
                    raise Exception("Failed to get response from OpenAI API")
            except Exception as e:
                print(f"Error generating response with OpenAI: {e}")
                return jsonify({
                    'success': False,
                    'error': f"OpenAI API error: {str(e)}"
                }), 400
                
        elif model_choice == 'gemini' and GENERATIVE_RAG_AVAILABLE:
            # Use Google Gemini API
            try:
                model_response = generate_gemini_response(user_message, relevant_chunks)
                if not model_response:
                    raise Exception("Failed to get response from Gemini API")
            except Exception as e:
                print(f"Error generating response with Gemini: {e}")
                return jsonify({
                    'success': False,
                    'error': f"Gemini API error: {str(e)}"
                }), 400
                
        else:
            # Use local StableBeluga model
            if llm is None:
                return jsonify({
                    'success': True,
                    'response': 'The language model is not available. Please check server logs or try using OpenAI or Gemini API instead.'
                })
                
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
            
            # Generate response with local model
            try:
                response = llm(
                    prompt,
                    max_tokens=1024,
                    temperature=0.7,
                    stop=["User:", "\n\n\n"]
                )
                model_response = response['choices'][0]['text'].strip()
            except Exception as e:
                print(f"Error generating response with local model: {e}")
                model_response = "I apologize, but I encountered an error while generating a response. Please try again later or try using OpenAI or Gemini API instead."
        
        return jsonify({
            'success': True,
            'response': model_response
        })
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
