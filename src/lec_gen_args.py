from pydantic import BaseModel, Field

class LecGenerateArgs(BaseModel):
    similarity_threshold: float = Field(..., description="The similarity threshold to merge similar images.")
    text_generating_context_size: int = Field(..., description="The context size for text generation, in the unit of slides.")
    max_tokens: int = Field(..., description="The maximum number of tokens for text generation.")
    pdf_name: str = Field(..., description="The name of the PDF file.")
    page_model: str = Field(..., description="The model name for generating the content of each page.")
    digest_model: str = Field(..., description="The model name for generating the introduction and summary.")
    tts_model: str = Field(..., description="The model name for generating the audio.")
    tts_voice: str = Field(..., description="The voice for generating the audio.")