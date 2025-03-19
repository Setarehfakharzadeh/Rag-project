from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from FlagEmbedding import FlagModel

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Gemini
GENAI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GENAI_API_KEY)
# Update model name to the latest version
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

# Initialize RAG components
embeddings_model_bge = FlagModel('BAAI/bge-base-en-v1.5', use_fp16=True)

# Load document chunks
MCP_CHUNKS_PATH = "mcp_chunks.json"
try:
    with open(MCP_CHUNKS_PATH, "r") as f:
        mcp_chunks = json.load(f)
        print(f"Loaded {len(mcp_chunks)} document chunks")
        
    # Generate embeddings
    document_texts = [chunk["content"] for chunk in mcp_chunks]
    embeddings_bge = embeddings_model_bge.encode(document_texts)
    print(f"Created embeddings for {len(document_texts)} chunks")
except Exception as e:
    print(f"Error loading document chunks: {e}")
    mcp_chunks = []
    embeddings_bge = np.array([])

PROMPT_TEMPLATE = """You are a Senior Software Developer at Microsoft. Use the context to answer the question. Give detailed answers.

Context:
{context}

Question: {question}

Answer:"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data['message']
    model_choice = data.get('model', 'local')  # Default to local model

    try:
        # RAG process
        emb_bge_query = embeddings_model_bge.encode([user_message])
        scores = np.dot(embeddings_bge, emb_bge_query.T).squeeze()
        
        # Get top chunks
        top_k = 3
        if len(scores) > 0:
            top_indices = np.argsort(-scores)[:top_k]
            relevant_chunks = [mcp_chunks[i] for i in top_indices]
        else:
            relevant_chunks = []
        
        # Format context from relevant chunks
        context = ""
        if relevant_chunks:
            context = "Here is some relevant information about Model Context Protocol:\n\n"
            for chunk in relevant_chunks:
                context += f"{chunk['content']}\n\n"
                context += f"(Source: {chunk['source']})\n\n"
        
        # Simple response for now (no LLM)
        if context:
            response = f"Here's what I found about your query: {user_message}\n\n{context}"
        else:
            response = f"I couldn't find specific information about: {user_message}"
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        print(f"Error in send_message endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data['message']
        model_choice = data.get('model', 'gemini')

        # RAG process
        query_embedding = embeddings_model_bge.encode([user_message])
        scores = np.dot(embeddings_bge, query_embedding.T).squeeze()
        
        # Get top 3 relevant chunks
        top_indices = np.argsort(-scores)[:3]
        context_chunks = [mcp_chunks[i]["content"] for i in top_indices]
        
        # Format context
        context_str = '\n###\n'.join(context_chunks)
        
        # Generate response
        if model_choice == 'gemini':
            prompt = PROMPT_TEMPLATE.format(
                context=context_str,
                question=user_message
            )
            response = gemini_model.generate_content(prompt)
            answer = response.text
        else:
            answer = "Currently only Gemini model is supported"

        return jsonify({
            'success': True,
            'response': answer
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5001)
