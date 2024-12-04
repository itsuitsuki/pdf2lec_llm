#!/bin/bash

# Check if file path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_pdf_file>"
    exit 1
fi

PDF_PATH="$1"

# Check if file exists
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File $PDF_PATH does not exist"
    exit 1
fi

# Make the API call
echo "Testing image merge with file: $PDF_PATH"
curl -X POST http://localhost:8000/api/v1/test_image_merge \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$PDF_PATH"

echo -e "\nDone!"