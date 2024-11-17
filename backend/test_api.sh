# from pydantic import BaseModel, Field

# class LecGenerateArgs(BaseModel):
#     similarity_threshold: float = Field(..., description="The similarity threshold to merge similar images.")
#     text_generating_context_size: int = Field(..., description="The context size for text generation, in the unit of slides.")
#     max_tokens: int = Field(..., description="The maximum number of tokens for text generation.")
#     pdf_name: str = Field(..., description="The name of the PDF file.")
#     page_model: str = Field(..., description="The model name for generating the content of each page.")
#     digest_model: str = Field(..., description="The model name for generating the introduction and summary.")
#     tts_model: str = Field(..., description="The model name for generating the audio.")
#     tts_voice: str = Field(..., description="The voice for generating the audio.")
#     complexity: int = Field(..., description="The complexity of the lecture.")
#     debug_mode: bool = Field(default=False, description="Enable debug mode to output detailed logs including generated text and prompts.")
#     use_rag: bool = Field(default=False, description="Whether to use RAG for text generation.")
#     textbook_name: str = Field(None, description="The name of the textbook PDF file. Must be provided if use_rag is True.")
#     openai_api_key: str = Field(None, description="OpenAI API key. If not provided, the API key from the environment variable OPENAI_API_KEY will be used.")

curl -X POST http://localhost:5000/api/v1/lec_generate \
-H "Content-Type: application/json" \
-d '{
    "similarity_threshold": 0.4,
    "text_generating_context_size": 2,
    "max_tokens": 1000,
    "pdf_name": "L6-Classsification-917_1-5",
    "page_model": "gpt-4o",
    "digest_model": "gpt-4o-mini",
    "tts_model": "tts-1",
    "tts_voice": "alloy",
    "complexity": 2,
    "debug_mode": true,
    "use_rag": true,
    "textbook_name": "Deep Learning Foundations and Concepts (Christopher M. Bishop, Hugh Bishop) (Z-Library)",
    "openai_api_key": null
}'

# curl -X POST "http://localhost:5000/api/v1/ask_question" \
# -H "Content-Type: application/json" \
# -d '{
#     "question": "What is a posterior probability?",
#     "task_id": "43ae4fe0-7e0a-4988-bf04-e6ec1f0978fe",
#     "max_tokens": 150,
#     "qa_model": "gpt-4-turbo",
#     "openai_api_key": null
# }'
