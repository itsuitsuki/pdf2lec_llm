# curl -X POST http://localhost:8000/api/v1/upload-slide \
#   -H "Content-Type: multipart/form-data" \
#   -F "file=@/home/ori/MyWorks/CS194/pdf2lec_llm/backend/test/cs168-fa24-lec14.pdf"

# curl http://localhost:8000/api/v1/pdfs

curl -X POST http://localhost:8000/api/v1/upload-textbook \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/home/ori/MyWorks/CS194/pdf2lec_llm/backend/test/Deep Learning Foundations and Concepts(.).pdf"

# curl http://localhost:8000/api/v1/pdfs