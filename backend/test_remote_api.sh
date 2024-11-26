curl -X POST http://18.224.173.161:5000/api/v1/lec_generate \
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
