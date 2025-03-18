export interface Message {
    content: string;
    isUser: boolean;
    timestamp: Date;
}

export interface ChatResponse {
    success: boolean;
    response?: string;
    error?: string;
} 