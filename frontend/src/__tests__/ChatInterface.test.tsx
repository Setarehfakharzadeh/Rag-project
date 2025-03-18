import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import ChatInterface from '../components/ChatInterface';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ChatInterface Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders the chat interface correctly', () => {
    render(<ChatInterface />);
    
    // Check for input field and send button
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    const sendButton = screen.getByText(/send/i);
    
    expect(inputElement).toBeInTheDocument();
    expect(sendButton).toBeInTheDocument();
  });

  test('handles user input correctly', () => {
    render(<ChatInterface />);
    
    const inputElement = screen.getByPlaceholderText(/type your message/i) as HTMLInputElement;
    
    // Type in the input field
    fireEvent.change(inputElement, { target: { value: 'Hello, AI!' } });
    
    expect(inputElement.value).toBe('Hello, AI!');
  });

  test('sends message when button is clicked', async () => {
    // Mock successful response
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        response: 'Hello, human!'
      }
    });
    
    render(<ChatInterface />);
    
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    const sendButton = screen.getByText(/send/i);
    
    // Type and send message
    fireEvent.change(inputElement, { target: { value: 'Hello, AI!' } });
    fireEvent.click(sendButton);
    
    // Check that axios was called with correct parameters
    expect(mockedAxios.post).toHaveBeenCalledWith('http://localhost:5001/api/chat', {
      message: 'Hello, AI!'
    });
    
    // Wait for the bot response to be rendered
    await waitFor(() => {
      const botResponse = screen.getByText('Hello, human!');
      expect(botResponse).toBeInTheDocument();
    });
  });

  test('displays error message when API call fails', async () => {
    // Mock failed response
    mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));
    
    render(<ChatInterface />);
    
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    const sendButton = screen.getByText(/send/i);
    
    // Type and send message
    fireEvent.change(inputElement, { target: { value: 'Hello, AI!' } });
    fireEvent.click(sendButton);
    
    // Wait for the error message to be rendered
    await waitFor(() => {
      const errorMessage = screen.getByText('Network error');
      expect(errorMessage).toBeInTheDocument();
    });
  });

  test('sends message when Enter key is pressed', async () => {
    // Mock successful response
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        success: true,
        response: 'Hello from Enter key!'
      }
    });
    
    render(<ChatInterface />);
    
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    
    // Type and press Enter
    fireEvent.change(inputElement, { target: { value: 'Enter key test' } });
    fireEvent.keyPress(inputElement, { key: 'Enter', code: 'Enter', charCode: 13 });
    
    // Check that axios was called with correct parameters
    expect(mockedAxios.post).toHaveBeenCalledWith('http://localhost:5001/api/chat', {
      message: 'Enter key test'
    });
    
    // Wait for the bot response to be rendered
    await waitFor(() => {
      const botResponse = screen.getByText('Hello from Enter key!');
      expect(botResponse).toBeInTheDocument();
    });
  });
}); 