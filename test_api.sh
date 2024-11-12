curl -X POST http://localhost:5000/lec_generate \
-H "Content-Type: application/json" \
-d '{
    "similarity_threshold": 0.7,
    "text_generating_context_size": 2,
    "max_tokens": 1000,
    "pdf_name": "L6-Classsification-917",
    "textbook_path": "./test/Deep Learning Foundations and Concepts (Christopher M. Bishop, Hugh Bishop) (Z-Library).pdf",
    "page_model": "gpt-4o",
    "digest_model": "gpt-4o-mini",
    "tts_model": "tts-1",
    "tts_voice": "alloy",
    "debug_mode": true,
    "complexity": 2
}'
