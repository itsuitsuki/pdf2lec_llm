from utils.encode_image_to_base64 import encode_image_path_to_base64
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm
import pymupdf
import cv2
import numpy as np
from pathlib import Path
import os
# from src.textbook_indexer import FAISSTextbookIndexer
from utils.similarity import calculate_similarity
from src.logger import CustomLogger
import logging
from prompts.multiagent_prompts import get_clarity_prompt, get_engagement_prompt, get_assembler_prompt
from prompts.slide_prompts import (
    get_each_slide_prompt, 
    get_summarizing_prompt, 
    get_introduction_prompt,
    get_slide_parsing_prompt  # Add this new import
)
# logger = CustomLogger.get_logger(__name__)
logger = logging.getLogger("uvicorn")
import json
import re


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
    base64_image = encode_image_path_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
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
        raise e

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Convert each page of a PDF file to an image and save them to the specified output directory.
    Also adds page numbers at the bottom of each image.
    """
    # Constants for page number rendering
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 2.0
    FONT_COLOR = (0, 0, 0)  # Black color
    FONT_THICKNESS = 3
    PADDING_BOTTOM = 60  # Space for page numbers

    # Create the output directory if it doesn't exist
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Open the PDF
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"Converting PDF to images: {total_pages} pages total")
    
    for page_num, page in enumerate(doc, 1):
        # Convert PDF page to image
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))
        
        # Convert PyMuPDF pixmap to numpy array (OpenCV format)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        
        if pix.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Add padding at bottom for page number
        img = cv2.copyMakeBorder(img, 0, PADDING_BOTTOM, 0, 0, 
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Add page number
        page_text = f"{page_num}/{total_pages}"
        text_size = cv2.getTextSize(page_text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = img.shape[0] - (PADDING_BOTTOM // 3)
        cv2.putText(img, page_text, (text_x, text_y), FONT, 
                   FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        # Save the image
        image_filename = f"page_{page_num}.png"
        cv2.imwrite(str(image_dir / image_filename), img)
        logger.info(f"Converted page {page_num}/{total_pages} to image")
    
    logger.info(f"PDF conversion completed: all {total_pages} pages converted to images")

def multiagent_generation(client, contents, model_name="gpt-4o", loopsize=1, max_tokens=500):
    """
    Perform multi-agent generation using clarity, engagement, and assembler agents to refine content.

    :param client: OpenAI API client
    :param contents: List of content (slides) to process
    :param clarity_prompt: Base prompt for the clarity agent
    :param engagement_prompt: Base prompt for the engagement agent
    :param assembler_prompt: Base prompt for the assembler agent
    :param model_name: The name of the model to use
    :param loopsize: Number of refinement loops for each content
    :param max_tokens: Maximum number of tokens to generate
    :return: List of refined content for each slide
    """

    clarity_prompt = get_clarity_prompt()
    engagement_prompt = get_engagement_prompt()
    assembler_prompt = get_assembler_prompt()

    refined_contents = []
    pbar = tqdm(total=len(contents))
    pbar.set_description("Refining slides")

    for i, content in enumerate(contents):
        current_content = content

        for _ in range(loopsize):
            # Clarity Agent
            clarity_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": clarity_prompt},
                    {"role": "user", "content": f"{clarity_prompt}\n\nContent:\n{current_content}"}
                ],
                max_tokens=max_tokens
            )
            clarity_critique = clarity_response.choices[0].message.content
            print(clarity_critique)

            # Engagement Agent
            engagement_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": engagement_prompt},
                    {"role": "user", "content": f"{engagement_prompt}\n\nContent:\n{current_content}"}
                ],
                max_tokens=max_tokens
            )
            engagement_critique = engagement_response.choices[0].message.content
            print(engagement_critique)

            # Assembler Agent
            assembler_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": assembler_prompt},
                    {
                        "role": "user",
                        "content": f"{assembler_prompt}\n\nContent:\n{current_content}\n\nClarity Critique:\n{clarity_critique}\n\nEngagement Critique:\n{engagement_critique}"
                    }
                ],
                max_tokens=max_tokens
            )
            current_content = assembler_response.choices[0].message.content

        refined_contents.append(current_content)
        pbar.update(1)

    pbar.close()
    return refined_contents

def merge_similar_images(image_dir, output_dir, similarity_threshold=0.4, max_merge=4):
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
    total_images = len(image_files)
    logger.info(f"Starting image merging process: {total_images} images to process")
    
    merged_groups = []
    current_group = [image_files[0]]
    
    for i in range(len(image_files) - 1):
        if len(current_group) >= max_merge:
            merged_groups.append(current_group)
            current_group = [image_files[i+1]]
            continue
            
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
    total_groups = len(merged_groups)
    logger.info(f"Merging {total_groups} groups of similar images")
    
    # Add these constants at the beginning of the function
    BORDER_SIZE = 10  # Size of black border in pixels
    
    for i, group in enumerate(merged_groups, 1):
        if len(group) == 1:
            merged = cv2.imread(os.path.join(image_dir, group[0]))
        else:
            images = [cv2.imread(os.path.join(image_dir, f)) for f in group]
            
            # Calculate target size
            max_height = max(img.shape[0] for img in images)
            max_width = max(img.shape[1] for img in images)
            
            if len(group) == 2:
                # Two images side by side with black border
                resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]
                merged = np.hstack([
                    resized_images[0],
                    np.zeros((max_height, BORDER_SIZE, 3), dtype=np.uint8),
                    resized_images[1]
                ])
                
            elif len(group) == 3:
                # Two images on top, one below (aligned with first image)
                resized_images = [cv2.resize(img, (max_width, max_height)) for img in images[:2]]
                # Create top row with first two images
                top_row = np.hstack([
                    resized_images[0],
                    np.zeros((max_height, BORDER_SIZE, 3), dtype=np.uint8),
                    resized_images[1]
                ])
                
                # Calculate the width of the third image to maintain aspect ratio
                third_img = images[2]
                aspect_ratio = third_img.shape[1] / third_img.shape[0]
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
                
                # Resize third image maintaining aspect ratio
                bottom_img = cv2.resize(third_img, (new_width, new_height))
                
                # Create bottom row with padding to match top row width
                padding_width = (top_row.shape[1] - new_width) // 2
                bottom_row = np.hstack([
                    np.zeros((max_height, padding_width, 3), dtype=np.uint8),
                    bottom_img,
                    np.zeros((max_height, top_row.shape[1] - new_width - padding_width, 3), dtype=np.uint8)
                ])
                
                # Stack top and bottom rows
                merged = np.vstack([
                    top_row,
                    np.zeros((BORDER_SIZE, top_row.shape[1], 3), dtype=np.uint8),
                    bottom_row
                ])
                
            elif len(group) == 4:
                # 2x2 grid with borders
                resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]
                top_row = np.hstack([
                    resized_images[0],
                    np.zeros((max_height, BORDER_SIZE, 3), dtype=np.uint8),
                    resized_images[1]
                ])
                bottom_row = np.hstack([
                    resized_images[2],
                    np.zeros((max_height, BORDER_SIZE, 3), dtype=np.uint8),
                    resized_images[3]
                ])
                merged = np.vstack([
                    top_row,
                    np.zeros((BORDER_SIZE, top_row.shape[1], 3), dtype=np.uint8),
                    bottom_row
                ])

        # Save merged image
        first_num = int(group[0].split('_')[1].split('.')[0])
        cv2.imwrite(os.path.join(output_dir, f'merged_{first_num:03d}.png'), merged)
        logger.info(f"Merged group {i}/{total_groups} ({len(group)} images)")
    
    logger.info(f"Image merging completed: {total_groups} merged images saved")

def generate_lecture_from_images_openai(
    client, 
    image_dir, 
    parsing_prompt, 
    faiss_textbook_indexer=None, 
    context_size=2, 
    model_name="gpt-4o", 
    max_tokens=500, 
    multiagent=False
    ):
    """
    Generate a complete lecture by analyzing images in sequence, maintaining context.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    total_slides = len(image_files)
    logger.info(f"Starting lecture generation: {total_slides} slides to process")
    
    context = []
    contents = []
    slide_analyses = []
    # 新增：存储RAG信息的列表
    rag_info = []
    
    pbar = tqdm(total=len(image_files))
    pbar.set_description("Generating lecture text")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # First pass: Parse the slide with structured output
        analysis_result = analyze_image_with_agent(
            client, 
            image_path, 
            parsing_prompt, 
            context[-context_size:] if context else [], 
            model_name=model_name, 
            max_tokens=max_tokens
        )
        slide_analyses.append(analysis_result)
        
        # Use the complexity from analysis to get the appropriate lecture prompt
        complexity = analysis_result['complexity']
        lecture_prompt = get_each_slide_prompt(complexity)
        
        # Get relevant textbook content using the keyword from analysis
        textbook_content = None
        current_rag_info = {
            "slide_number": i + 1,
            "keyword": analysis_result['keyword'],
            "textbook_content": None
        }
        
        if faiss_textbook_indexer:
            query = analysis_result['keyword']
            textbook_content = faiss_textbook_indexer.get_relevant_content(
                query=query,
                index_name=Path(faiss_textbook_indexer.textbook_path).stem
            )
            current_rag_info["textbook_content"] = textbook_content
            logger.debug(f"Textbook content: {textbook_content}")
        
        rag_info.append(current_rag_info)
        
        # Build the context prompt with the new structure
        context_prompt = (
            f"{lecture_prompt}\n\n"
            f"Context:\n{' '.join(context[-context_size:]) if context else 'No previous context'}\n\n"
        )
        
        if textbook_content:
            context_prompt += "Relevant textbook information provided for you:\n" + "\n".join(textbook_content) + "\n\n"
            
        context_prompt += (
            f"Here is the explanation of the image provided for you:\n"
            f"{analysis_result['description']}\n\n"
        )
        
        # Use the complexity from analysis instead of user input
        slide_content = analyze_page_by_image_openai(
            client, 
            context_prompt, 
            image_path, 
            model_name=model_name, 
            max_tokens=max_tokens
        )
        
        context.append(slide_content)
        contents.append(slide_content)
        if len(context) > context_size:
            context.pop(0)
        
        logger.info(f"Completed slide {i}/{total_slides}")
    
    logger.info(f"Lecture generation completed: all {total_slides} slides processed")
    
    # 保存RAG信息到JSON文件
    base_dir = str(Path(image_dir).parent)
    rag_info_path = os.path.join(base_dir, "rag_info.json")
    with open(rag_info_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_slides": total_slides,
            "rag_details": rag_info
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"RAG information saved to {rag_info_path}")
    
    if multiagent:
        logger.info("Starting multi-agent refinement")
        contents = multiagent_generation(client, contents, model_name=model_name)
        logger.info("Multi-agent refinement completed")
    
    return contents, image_files, slide_analyses

def digest_lecture_openai(client, complete_lecture, digest_prompt, model_name="gpt-4o-mini"):
    """Generate lecture summary"""
    logger.info("Starting lecture summary generation")
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
    logger.info("Lecture summary generation completed")
    return summary.choices[0].message.content

def analyze_image_with_agent(client, image_path, prompt, context, model_name="gpt-4o", max_tokens=500):
    """
    Analyze an image and return a detailed description, page number, keyword, and complexity.

    :param client: OpenAI API client
    :param image_path: Local path to the image file
    :param prompt: The text prompt for the model
    :param context: Context from the previous image analysis
    :param model_name: The name of the model to use
    :param max_tokens: The maximum number of tokens to generate
    :return: A dictionary with description, page number, keyword, and complexity
    """
    # Encode the image
    base64_image = encode_image_path_to_base64(image_path)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": f"Context: {context}"}
                    ]
                }
            ],
            max_tokens=max_tokens
        )
        # Parse the response to extract the required fields
        content = response.choices[0].message.content
        
        # 首先尝试匹配markdown代码块中的JSON
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有markdown代码块，尝试直接匹配最外层的花括号内容
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No valid JSON found in response")

        try:
            parsed_response = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw response content: {content}")
            logger.error(f"Extracted JSON string: {json_str}")
            # 提供默认响应
            parsed_response = {
                "Description": "Failed to parse slide content",
                "Keywords": "error",
                "Complexity": 1
            }
        
        # 构建结果，包含默认值
        result = {
            "description": parsed_response.get("Description", "No description available"),
            "keyword": parsed_response.get("Keywords", "No keywords available"),
            "complexity": int(parsed_response.get("Complexity", 1))
        }
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        logger.error(f"Raw response content: {content if 'content' in locals() else 'No content generated'}")
        raise e

def get_slide_parsing_prompt():
    """Load the slide parsing prompt from file"""
    prompt_path = Path("backend/prompts/slide_parsing_agent.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()