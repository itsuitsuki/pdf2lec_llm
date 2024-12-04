curl -X POST http://localhost:5000/api/v1/lec_generate \
-H "Content-Type: application/json" \
-d '{
    "similarity_threshold": 0.4,
    "text_generating_context_size": 2,
    "max_tokens": 1000,
    "pdf_name": "random_document.pdf",
    "page_model": "gpt-4o",
    "digest_model": "gpt-4o-mini",
    "tts_model": "tts-1",
    "tts_voice": "alloy",
    "complexity": 2,
    "debug_mode": true,
    "use_rag": false,
    "openai_api_key": null
}'