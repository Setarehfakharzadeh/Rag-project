from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple response function for testing
def get_answer(question):
    return {"success": True, "response": f"You said: {question}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
            
        user_message = data['message']
        result = get_answer(user_message)
        
        return jsonify(result)
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Debug logging
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Flask server...")  # Debug logging
    app.run(debug=True, port=5001)
