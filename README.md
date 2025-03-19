A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

**Author**: [Setarehfakharzadeh](https://github.com/Setarehfakharzadeh)

## 🤖 About the Project
# RAG Chat Interface

A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

## 🤖 About the Project

This project combines a modern React frontend with a Flask backend to create an intelligent chat interface powered by the StableBeluga-7B language model. The system uses RAG (Retrieval-Augmented Generation) to enhance responses by processing and retrieving information from the Model Context Protocol (MCP) documentation.

### RAG Implementation Details

The RAG system works by:
1. Processing and chunking Model Context Protocol documentation into smaller, manageable pieces
2. Using the all-MiniLM-L6-v2 sentence transformer model to create embeddings
3. Implementing efficient document retrieval based on semantic similarity
4. Enhancing the StableBeluga-7B responses with relevant context from the documentation

### Multiple Model Support

The project supports three different language model backends:

1. **StableBeluga-7B** (Default): A locally-run 7B parameter model that runs entirely on your machine without requiring external API calls.

2. **OpenAI GPT Models**: An alternative implementation that uses OpenAI's API (GPT-3.5-Turbo/GPT-4) for generating responses. This option requires:
   - An OpenAI API key
   - Internet connectivity
   - Setting up environment variables

3. **Google Gemini API**: Another cloud-based alternative that uses Google's Gemini Pro model. This option requires:
   - A Google Gemini API key
   - Internet connectivity
   - Setting up environment variables

### Data Processing
- Source: Model Context Protocol (MCP) documentation, including:
  - Server quickstart guide
  - Client implementation guide
  - Latest specifications
- Document Processing: Content is split into optimal chunks of up to 512 characters
- Embedding Model: all-MiniLM-L6-v2 for efficient semantic search
- Storage: Processed documents and embeddings are indexed for fast retrieval

### Key Features

- **Large Language Model Integration**: Utilizes StableBeluga-7B (4GB quantized model) for generating responses
- **OpenAI Integration**: Alternative implementation using OpenAI's GPT models
- **Google Gemini API Integration**: Alternative implementation using Google's Gemini Pro model
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
- OpenAI API integration (optional)
- Google Gemini API integration (optional)
- RAG implementation for context-aware responses
- Sentence Transformers for embeddings
- Web crawling capabilities for MCP documentation

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- 8GB+ RAM recommended
- OpenAI API key (optional, for using OpenAI models)
- Google Gemini API key (optional, for using Google Gemini API)

### Language Model Setup
Due to GitHub's file size limitations, the language model file (`stablebeluga-7b.Q4_K_M.gguf`) is not included in this repository. To use this project:

1. Download the model:
   ```bash
   # Download from Hugging Face
   wget https://huggingface.co/TheBloke/StableBeluga-7B-GGUF/resolve/main/stablebeluga-7b.Q4_K_M.gguf
   ```

2. Place the downloaded file in the project root directory.

### OpenAI API Setup (Optional)
To use the OpenAI integration:

1. Create a `.env` file in the backend directory:
   ```bash
   cd backend
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

2. Install the OpenAI Python package:
   ```bash
   pip install openai
   ```

### Google Gemini API Setup (Optional)
To use the Google Gemini API integration:

1. Create a `.env` file in the backend directory:
   ```bash
   cd backend
   echo "GOOGLE_GEMINI_API_KEY=your_api_key_here" > .env
   ```

2. Install the Google Gemini API Python package:
   ```bash
   pip install google-gemini-api
   ```

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

### Using OpenAI Models (Optional)
To run the application with OpenAI models instead of StableBeluga:

```bash
cd backend
python generative_rag.py
```

### Using Google Gemini API (Optional)
To run the application with Google Gemini API instead of StableBeluga:

```bash
cd backend
python google_gemini_rag.py
```

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
│   ├── app.py              # Main application file (StableBeluga)
│   ├── generative_rag.py   # OpenAI integration for RAG
│   ├── google_gemini_rag.py # Google Gemini API integration for RAG
│   ├── direct_openai_rag.py # Direct OpenAI API implementation
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
- Using OpenAI models requires an API key and internet connectivity
- Using Google Gemini API requires an API key and internet connectivity

## 🔗 Links
- [GitHub Repository](https://github.com/Setarehfakharzadeh/Rag-project)
- [Issues](https://github.com/Setarehfakharzadeh/Rag-project/issues)
- [StableBeluga-7B Model](https://huggingface.co/TheBloke/StableBeluga-7B-GGUF)
- [Model Context Protocol](https://modelcontextprotocol.io) 
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Google Gemini API Documentation](https://cloud.google.com/gemini/docs)