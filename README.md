# RAG Chat Interface

A sophisticated chat interface implementing Retrieval-Augmented Generation (RAG) with a large language model for enhanced conversational AI capabilities.

## 🤖 About the Project

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
- Git LFS (for handling large model files)
- 8GB+ RAM recommended

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

### Language Model
The project uses StableBeluga-7B (4GB quantized version). The model file is tracked using Git LFS due to its size.

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
├── .github/              # GitHub Actions workflows
└── stablebeluga-7b.Q4_K_M.gguf  # Language model file (Git LFS)
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Important Notes
- The language model file (stablebeluga-7b.Q4_K_M.gguf) is large (~4GB) and is handled using Git LFS
- Ensure sufficient RAM is available when running the application
- The model may take a few moments to load on first startup

## 🔗 Links
- [GitHub Repository](https://github.com/Setarehfakharzadeh/Rag-project)
- [Issues](https://github.com/Setarehfakharzadeh/Rag-project/issues) 