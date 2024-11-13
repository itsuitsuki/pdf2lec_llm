import io
from PIL import Image
import pymupdf

def extract_elements_from_pdf(pdf_path: str):
    """Extracts both text and non-text elements from the PDF."""
    elements = {"text": "", "images": []}
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            # Extract text
            elements["text"] += page.get_text()

            # Extract images
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                elements["images"].append(image)
    
    return elements