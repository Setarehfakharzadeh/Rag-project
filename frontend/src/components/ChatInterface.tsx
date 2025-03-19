import React, { useState, useEffect, useRef } from 'react';
import { Message, ChatResponse } from '../types/chat';
import axios from 'axios';
import './ChatInterface.css';

type ModelType = 'local' | 'openai' | 'gemini';

const ChatInterface: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState<ModelType>('local');
    const chatBoxRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        if (chatBoxRef.current) {
            chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
        }
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            content: input,
            isUser: true,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await axios.post<ChatResponse>('http://localhost:5001/api/chat', {
                message: input,
                model: selectedModel
            });

            if (response.data.success && response.data.response) {
                const botMessage: Message = {
                    content: response.data.response,
                    isUser: false,
                    timestamp: new Date()
                };
                setMessages(prev => [...prev, botMessage]);
            } else {
                throw new Error(response.data.error || 'Failed to get response');
            }
        } catch (error) {
            const errorMessage: Message = {
                content: error instanceof Error ? error.message : 'An error occurred',
                isUser: false,
                timestamp: new Date()
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setSelectedModel(e.target.value as ModelType);
    };

    return (
        <div className="chat-container">
            <div className="model-selector">
                <label htmlFor="model-select">Select Model: </label>
                <select 
                    id="model-select" 
                    value={selectedModel} 
                    onChange={handleModelChange}
                    disabled={isLoading}
                >
                    <option value="local">StableBeluga (Local)</option>
                    <option value="openai">OpenAI GPT</option>
                    <option value="gemini">Google Gemini</option>
                </select>
            </div>
            <div className="chat-box" ref={chatBoxRef}>
                {messages.map((message, index) => (
                    <div
                        key={index}
                        className={`message ${message.isUser ? 'user-message' : 'bot-message'}`}
                    >
                        <div className="message-content">{message.content}</div>
                        <div className="message-timestamp">
                            {message.timestamp.toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="message bot-message">
                        <div className="loading-dots">
                            <span>.</span><span>.</span><span>.</span>
                        </div>
                    </div>
                )}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message..."
                    disabled={isLoading}
                />
                <button 
                    onClick={sendMessage}
                    disabled={isLoading || !input.trim()}
                >
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};

export default ChatInterface;