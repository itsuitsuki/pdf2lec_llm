export interface PDFFile {
    id: string;
    filename: string;
    path: string;
    type: 'slide' | 'textbook';
    status?: 'pending' | 'completed' | 'failed';
}
  
export interface TaskResponse {
    task_id: string;
    status: 'pending' | 'completed' | 'failed';
    metadata?: any;
    error?: string;
}

export interface GenerateOptions {
    similarity_threshold: number;
    text_generating_context_size: number;
    max_tokens: number;
    pdf_name: string;
    page_model: string;
    digest_model: string;
    tts_model: string;
    tts_voice: string;
    complexity: number;
    debug_mode: boolean;
    use_rag: boolean;
    textbook_name: string | null;
}