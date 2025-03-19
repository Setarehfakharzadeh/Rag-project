export interface Message {
    content: string;
    isUser: boolean;
    timestamp: Date;
}

export type ModelType = 'local' | 'openai' | 'gemini';

export interface ChatResponse {
    success: boolean;
    response?: string;
    error?: string;
} 