A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

**Author**: [Setarehfakharzadeh](https://github.com/Setarehfakharzadeh)

## 🤖 About the Project
# RAG Chat Interface

A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

## 🤖 About the Project

This project combines a modern React frontend with a Flask backend to create an intelligent chat interface powered by the StableBeluga-7B language model. The system uses RAG (Retrieval-Augmented Generation) to provide more accurate and context-aware responses.

### Key Features

- **Large Language Model Integration**: Utilizes StableBeluga-7B (4GB quantized model) for generating responses
- **RAG Implementation**: Enhanced response generation with context retrieval
- **Model Context Protocol Integration**: Specialized knowledge base for MCP documentation
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
- Sentence Transformers for embeddings
- Web crawling capabilities for MCP documentation

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

### Crawl MCP Documentation (Optional)
To create the RAG knowledge base for the Model Context Protocol:

```bash
cd backend
python crawl_mcp.py
```

This will:
1. Crawl the MCP documentation website
2. Extract and process content
3. Generate embeddings for RAG
4. Save data for use by the chat interface

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

### RAG Testing
To test the retrieval capabilities with MCP documentation:

```bash
cd backend
python test_rag.py
```

This script tests the retrieval of relevant MCP documentation for sample queries.

### Frontend Tests
The frontend includes comprehensive test coverage using Jest and React Testing Library:

```bash
# Navigate to the frontend directory
cd frontend

# Run tests
npm test

# Run tests without watch mode
npm test -- --watchAll=false

# Or use the provided script
./run-tests.sh
```

The frontend tests cover:
- Component rendering
- User interactions (typing, clicking buttons)
- API integration
- Error handling

### Backend Tests
```bash
# Backend tests
cd backend
python -m pytest tests/
```

## 📝 Project Structure
```
.
├── frontend/                # React frontend application
│   ├── src/                # Source files
│   │   ├── __tests__/     # Frontend tests
│   ├── public/             # Public assets
│   └── package.json        # Dependencies
├── backend/                # Flask backend
│   ├── app.py              # Main application file
│   ├── crawl_mcp.py        # Web crawler for MCP docs
│   ├── test_rag.py         # RAG testing utilities
│   ├── templates/          # HTML templates
│   ├── static/             # Static files
│   └── requirements.txt    # Python dependencies
├── .devcontainer/          # Development container configuration
└── .github/                # GitHub Actions workflows
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes
- The language model file is large (~4GB) and must be downloaded separately
- MCP documentation crawling requires internet access
- Ensure sufficient RAM is available when running the application
- The model may take a few moments to load on first startup

## 🔗 Links
- [GitHub Repository](https://github.com/Setarehfakharzadeh/Rag-project)
- [Issues](https://github.com/Setarehfakharzadeh/Rag-project/issues)
- [StableBeluga-7B Model](https://huggingface.co/TheBloke/StableBeluga-7B-GGUF)
- [Model Context Protocol](https://modelcontextprotocol.io) 