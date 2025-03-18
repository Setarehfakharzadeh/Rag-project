# RAG Chat Interface

A chat interface built with React frontend and Flask backend, using the TinyLlama model for responses.

## Project Structure

```
Rag final/
├── backend/         # Flask backend
├── frontend/        # React frontend
└── README.md       # This file
```

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the Flask server:
```bash
python -m flask run --port 5001
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the React development server:
```bash
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5001

## Features

- Real-time chat interface
- AI-powered responses using TinyLlama model
- Modern React frontend with TypeScript
- RESTful Flask backend
- Clean and responsive UI

## Dependencies

### Backend
- Flask
- Flask-CORS
- Transformers
- PyTorch
- Accelerate

### Frontend
- React
- TypeScript
- Axios
- React Scripts

## Development

To modify the application:
1. Backend code is in `backend/app.py`
2. Frontend components are in `frontend/src/components/`
3. Styles are in corresponding `.css` files

## Notes

- The backend uses TinyLlama, a smaller language model suitable for testing and development
- Make sure all dependencies are installed before running the application
- The frontend and backend must both be running for the application to work properly 