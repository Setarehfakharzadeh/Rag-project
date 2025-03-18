import React from 'react';
import { render, screen } from '@testing-library/react';
import App from '../App';

// Mock the ChatInterface component to isolate tests
jest.mock('../components/ChatInterface', () => {
  return function MockChatInterface() {
    return <div data-testid="mock-chat-interface">Chat Interface Mock</div>;
  };
});

describe('App Component', () => {
  test('renders the app header', () => {
    render(<App />);
    const headerElement = screen.getByText(/RAG Chat Interface/i);
    expect(headerElement).toBeInTheDocument();
  });

  test('renders the chat interface component', () => {
    render(<App />);
    const chatInterfaceElement = screen.getByTestId('mock-chat-interface');
    expect(chatInterfaceElement).toBeInTheDocument();
  });

  test('has the correct layout structure', () => {
    const { container } = render(<App />);
    
    // Check that App contains a header and main elements
    const header = container.querySelector('header');
    const main = container.querySelector('main');
    
    expect(header).toBeInTheDocument();
    expect(main).toBeInTheDocument();
  });
}); 