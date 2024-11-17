curl -X POST http://localhost:5000/api/v1/split_merge_pdf \
-H "Content-Type: application/json" \
-d '{
    "pdf_name": "cs168-fa24-lec14",
    "similarity_threshold": 0.4,
    "debug_mode": false
}'