from pydantic import BaseModel, Field
from typing import Optional
class LecGenerateArgs(BaseModel):
    similarity_threshold: float = Field(..., description="The similarity threshold to merge similar images.")
    text_generating_context_size: int = Field(..., description="The context size for text generation, in the unit of slides.")
    max_tokens: int = Field(..., description="The maximum number of tokens for text generation.")
    pdf_name: str = Field(..., description="The name of the PDF file.")
    page_model: str = Field(..., description="The model name for generating the content of each page.")
    digest_model: str = Field(..., description="The model name for generating the introduction and summary.")
    tts_model: str = Field(..., description="The model name for generating the audio.")
    tts_voice: str = Field(..., description="The voice for generating the audio.")
    complexity: int = Field(..., description="The complexity of the lecture.")
    debug_mode: bool = Field(default=False, description="Enable debug mode to output detailed logs including generated text and prompts.")
    use_rag: bool = Field(default=False, description="Whether to use RAG for text generation.")
    textbook_name: Optional[str] = Field(None, description="The name of the textbook PDF file. Must be provided if use_rag is True.")
    multiagent: bool = Field(default=False, description="Whether to use multi-agent prompts for text generation.") # TODO: Include this into the documentation
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key. If not provided, the API key from the environment variable OPENAI_API_KEY will be used.")
    
class QAArgs(BaseModel):
    question: str = Field(..., description="The question to ask.")
    task_id: str = Field(..., description="The task ID.")
    max_tokens: int = Field(..., description="The maximum number of tokens for text generation.")
    qa_model: str = Field(..., description="The model name for generating the answer.")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key. If not provided, the API key from the environment variable OPENAI_API_KEY will be used.")
    use_rag: bool = Field(default=False, description="Whether to use RAG for text generation.")
    textbook_name: Optional[str] = Field(None, description="The name of the textbook PDF file. Must be provided if use_rag is True.")

class PDFSplitMergeArgs(BaseModel):
    pdf_name: str = Field(..., description="The name of the PDF file.")
    similarity_threshold: float = Field(default=0.4, description="The similarity threshold to merge similar images.")
    debug_mode: bool = Field(default=False, description="Enable debug logging.")