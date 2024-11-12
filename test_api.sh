curl -X POST http://localhost:5000/lec_generate \
-H "Content-Type: application/json" \
-d '{
    "similarity_threshold": 0.4,
    "text_generating_context_size": 2,
    "max_tokens": 1000,
    "pdf_name": "handout4_binary_hypothesis_testing",
    "page_model": "gpt-4o",
    "digest_model": "gpt-4o-mini",
    "tts_model": "tts-1",
    "tts_voice": "alloy"
    "complexity": 2
}'
