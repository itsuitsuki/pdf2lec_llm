export interface PDFFile {
    id: string;
    filename: string;
    path: string;
    type: 'slide' | 'textbook';
}
  
export interface TaskResponse {
    task_id: string;
    status: 'pending' | 'completed' | 'failed';
    metadata?: any;
    error?: string;
}

export interface GenerateLectureOptions {
    debug_mode?: boolean;
    use_rag?: boolean;
    textbook_name?: string;
}