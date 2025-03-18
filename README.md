# RAG Chat Interface

A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

**Author**: [Setarehfakharzadeh](https://github.com/Setarehfakharzadeh)

##  About the Project

This project combines a modern React frontend with a Flask backend to create an intelligent chat interface powered by the StableBeluga-7B language model. The system uses RAG (Retrieval-Augmented Generation) to provide more accurate and context-aware responses.

### Key Features

- **Large Language Model Integration**: Utilizes StableBeluga-7B (4GB quantized model) for generating responses
- **RAG Implementation**: Enhanced response generation with context retrieval
- **Modern Tech Stack**: React frontend with TypeScript + Flask backend
- **Real-time Chat Interface**: Responsive and user-friendly design
- **DevContainer Support**: Easy development environment setup

## 🛠️ Technology Stack

### Frontend
- React with TypeScript
- Modern CSS for styling
- Real-time communication with backend

### Backend
- Flask (Python)
- StableBeluga-7B Language Model
- RAG implementation for context-aware responses

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- 8GB+ RAM recommended

### Language Model Setup
Due to GitHub's file size limitations, the language model file (`stablebeluga-7b.Q4_K_M.gguf`) is not included in this repository. To use this project:

1. Download the model:
   ```bash
   # Download from Hugging Face
   wget https://huggingface.co/TheBloke/StableBeluga-7B-GGUF/resolve/main/stablebeluga-7b.Q4_K_M.gguf
   ```

2. Place the downloaded file in the project root directory.

### Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

## 🚀 Running the Application

### Start Backend
```bash
cd backend
python app.py
```
The backend will start on http://localhost:5001

### Start Frontend
```bash
cd frontend
npm start
```
The frontend will be available at http://localhost:3000

## 🧪 Testing
```bash
# Backend tests
cd backend
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

## 📝 Project Structure
```
.
├── frontend/                # React frontend application
│   ├── src/                # Source files
│   ├── public/             # Public assets
│   └── package.json        # Dependencies
├── backend/                # Flask backend
│   ├── app.py             # Main application file
│   ├── templates/         # HTML templates
│   ├── static/           # Static files
│   └── requirements.txt   # Python dependencies
├── .devcontainer/         # Development container configuration
└── .github/              # GitHub Actions workflows
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes
- The language model file is large (~4GB) and must be downloaded separately
- Ensure sufficient RAM is available when running the application
- The model may take a few moments to load on first startup

## 🔗 Links
- [GitHub Repository](https://github.com/Setarehfakharzadeh/Rag-project)
- [Issues](https://github.com/Setarehfakharzadeh/Rag-project/issues)
- [StableBeluga-7B Model](https://huggingface.co/TheBloke/StableBeluga-7B-GGUF)

## RAG Implementation

This project includes a Retrieval-Augmented Generation (RAG) implementation to demonstrate how to use external documentation to enhance AI responses. The RAG system currently works with Model Context Protocol (MCP) documentation.

### RAG Components

1. **Document Fetcher** (`mcp_fetch.py`): A script to fetch and preprocess MCP documentation from the official website. It extracts content from:
   - Server quickstart guide
   - Client quickstart guide
   - MCP specifications
   
   The script cleans the HTML content, splits it into manageable chunks, and stores them in a JSON file.

2. **Simple RAG Demo** (`rag_test.py`): An interactive demo that uses keyword-based search to find relevant information from the documentation and displays it to the user.

3. **Advanced RAG Demo** (`advanced_rag_test.py`): A more complex implementation that not only retrieves relevant chunks but also uses the Llama model to generate coherent responses based on the retrieved information.

### Running the RAG Demo

1. First, fetch the MCP documentation:
   ```
   cd backend
   python mcp_fetch.py
   ```

2. Run the simple RAG demo:
   ```
   python rag_test.py
   ```
   
3. Or try the advanced demo with the Llama model:
   ```
   python advanced_rag_test.py
   ```

The simple demo uses keyword matching to find relevant documentation, while the advanced demo adds LLM-generated responses based on the retrieved content.

### How it Works

1. **Retrieval**: When you ask a question, the system finds the most relevant chunks from the MCP documentation.
2. **Augmentation**: These chunks provide context for answering your question.
3. **Generation**: For the advanced demo, the Llama model uses this context to generate a comprehensive answer.

This approach helps the model provide more accurate and detailed answers about MCP than it could with its training data alone. 