from utils.encode_image_to_base64 import encode_image_to_base64
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pymupdf
import cv2
import numpy as np
from pathlib import Path
import os
from src.textbook_indexer import TextbookIndexer

def analyze_page_by_image_openai(client, prompt, image_path, textbook_content=None, model_name="gpt-4o", max_tokens=500):
    """
    Analyze an image using the GPT-4o model and return a description.

    :param client: OpenAI API client
    :param prompt: The text prompt for the model
    :param image_path: Local path to the image file
    :param textbook_content: Relevant textbook content for enhanced generation
    :param model_name: The name of the model to use
    :param max_tokens: The maximum number of tokens to generate
    :return: Model-generated description or error message
    """
    
    # Encode the image
    base64_image = encode_image_to_base64(image_path)
    
    # Modify prompt to include textbook content if available
    enhanced_prompt = prompt
    if textbook_content:
        enhanced_prompt += "\n\nRelevant textbook content:\n" + "\n".join(textbook_content)
        enhanced_prompt += "\n\nPlease incorporate relevant information from the textbook in your explanation."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Convert each page of a PDF file to an image and save them to the specified output directory.

    :param pdf_path: Path to the PDF file
    :param output_dir: Directory to save the converted images
    """
    # Create the output directory if it doesn't exist
    # pdf_name = Path(pdf_path).stem
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Open the PDF
    doc = pymupdf.open(pdf_path)

    # Iterate through each page
    for page_num, page in enumerate(doc):
        # Convert the page to an image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))  # 300 DPI
        
        # Save the image
        image_filename = f"page_{page_num+1}.png"
        pix.save(image_dir / image_filename)
    # print(f"PDF pages converted to images and saved to {output_dir}")
        
def calculate_similarity(img1, img2):
    """
    Calculate the similarity between two images using ORB feature matching.
    This method is invariant to translation and rotation.
    
    :param img1: First image
    :param img2: Second image
    :return: A similarity score between 0 and 1
    """
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score
    similarity = len(matches) / max(len(kp1), len(kp2))
    
    return similarity

def merge_similar_images(image_dir, output_dir, similarity_threshold=0.7):
    """
    Merge similar consecutive images in a directory while maintaining the original order.
    
    :param image_dir: Directory containing the images
    :param output_dir: Directory to save the merged images
    :param similarity_threshold: Threshold for considering images as similar
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all image files sorted by name
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    merged_groups = []
    current_group = [image_files[0]]
    
    for i in range(len(image_files) - 1):
        img1 = cv2.imread(os.path.join(image_dir, image_files[i]))
        img2 = cv2.imread(os.path.join(image_dir, image_files[i+1]))
        
        similarity = calculate_similarity(img1, img2)
        
        if similarity >= similarity_threshold:
            current_group.append(image_files[i+1])
        else:
            merged_groups.append(current_group)
            current_group = [image_files[i+1]]
    
    # Add the last group
    if current_group:
        merged_groups.append(current_group)
    
    # Merge and save images
    for i, group in enumerate(merged_groups):
        if len(group) == 1:
            img = cv2.imread(os.path.join(image_dir, group[0]))
            merged = img
        else:
            images = [cv2.imread(os.path.join(image_dir, f)) for f in group]
            heights = [img.shape[0] for img in images]
            max_width = max(img.shape[1] for img in images)
            merged = np.vstack([cv2.resize(img, (max_width, img.shape[0])) for img in images])
        
        # Use the first image's number in the group for naming
        first_num = int(group[0].split('_')[1].split('.')[0])
        cv2.imwrite(os.path.join(output_dir, f'merged_{first_num:03d}.png'), merged)
        # :03d means 3 digits with leading zeros, equivalent to zfill(3)
    
    # print(f"Merged images saved to {output_dir}")
    
def generate_lecture_from_images_openai(client, image_dir, prompt, textbook_indexer=None, context_size=2, model_name="gpt-4o", max_tokens=500):
    """
    Generate a complete lecture by analyzing images in sequence, maintaining context.
    
    :param client: OpenAI API client
    :param image_dir: Directory containing the merged images
    :param prompt: The base prompt to use for image analysis
    :param textbook_indexer: Textbook indexer object
    :param context_size: Number of previous slides to consider for context
    :param model_name: The name of the model to use
    :param max_tokens: The maximum number of tokens to generate
    :return: A list of content for each slide, and the list of image files corresponding to each slide
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    # full_lecture = ""
    context = []
    contents = []
    pbar = tqdm(total=len(image_files))
    pbar.set_description("Generating lecture text")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # Get relevant textbook content if available
        textbook_content = None
        if textbook_indexer:
            # Extract text from image for query (you might need to implement this)
            # For now, we'll use the context as query
            query = " ".join(context[-2:]) if context else "Introduction to the topic"
            textbook_content = textbook_indexer.get_relevant_content(
                query=query,
                index_name=Path(textbook_indexer.textbook_path).stem
            )
        
        context_prompt = f"{prompt}\n\nContext from previous slides:\n{' '.join(context)}\n\nAnalyze the current slide in the context of what has been discussed before. Remember do not repeat the same information."
        
        slide_content = analyze_page_by_image_openai(
            client, 
            context_prompt, 
            image_path, 
            textbook_content=textbook_content,
            model_name=model_name, 
            max_tokens=max_tokens
        )
        
        # Update context
        context.append(slide_content)
        contents.append(slide_content)
        if len(context) > context_size:
            context.pop(0)
        pbar.update(1)
        pbar.set_postfix_str(f"Slide {i+1} analyzed")
        
    pbar.close()
    return contents, image_files

def digest_lecture_openai(client, complete_lecture, digest_prompt, model_name="gpt-4o-mini"):
    summary = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": complete_lecture + '\n\n' + digest_prompt
            }
        ]
    )

    return summary.choices[0].message.content