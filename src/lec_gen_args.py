from pydantic import BaseModel, Field

class LecGenerateArgs(BaseModel):
    similarity_threshold: float = Field(..., description="The similarity threshold to merge similar images.")
    debug_mode: bool = Field(default=False, description="Enable debug mode to output detailed logs including generated text and prompts.")
    text_generating_context_size: int = Field(..., description="The context size for text generation, in the unit of slides.")
    max_tokens: int = Field(..., description="The maximum number of tokens for text generation.")
    pdf_name: str = Field(..., description="The name of the PDF file.")
    textbook_path: str = Field(None, description="Path to the textbook PDF file (optional)")
    page_model: str = Field(..., description="The model name for generating the content of each page.")
    digest_model: str = Field(..., description="The model name for generating the introduction and summary.")
    tts_model: str = Field(..., description="The model name for generating the audio.")
    tts_voice: str = Field(..., description="The voice for generating the audio.")