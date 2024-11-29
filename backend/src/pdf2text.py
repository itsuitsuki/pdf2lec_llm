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
# logger = CustomLogger.get_logger(__name__)
logger = logging.getLogger("uvicorn")

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

    :param pdf_path: Path to the PDF file
    :param output_dir: Directory to save the converted images
    """
    # Create the output directory if it doesn't exist
    # pdf_name = Path(pdf_path).stem
    image_dir = Path(output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Open the PDF
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    logger.info(f"Converting PDF to images: {total_pages} pages total")
    
    for page_num, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(300/72, 300/72))
        image_filename = f"page_{page_num}.png"
        pix.save(image_dir / image_filename)
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
    
    for i, group in enumerate(merged_groups, 1):
        if len(group) == 1:
            merged = cv2.imread(os.path.join(image_dir, group[0]))
        else:
            images = [cv2.imread(os.path.join(image_dir, f)) for f in group]
            
            # 计算目标尺寸
            max_height = max(img.shape[0] for img in images)
            max_width = max(img.shape[1] for img in images)
            
            if len(group) == 2:
                # 两张图片横向排列
                resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]
                merged = np.hstack(resized_images)
                
            elif len(group) == 3:
                # 前两张横向，第三张在下方
                img1, img2, img3 = images
                # 调整前两张图片大小并横向合并
                resized_top = [cv2.resize(img, (max_width, max_height)) for img in [img1, img2]]
                top_row = np.hstack(resized_top)
                # 调整第三张图片大小以匹配上方宽度
                bottom_img = cv2.resize(img3, (top_row.shape[1], max_height))
                merged = np.vstack([top_row, bottom_img])
                
            elif len(group) == 4:
                # 2x2 网格排列
                resized_images = [cv2.resize(img, (max_width, max_height)) for img in images]
                top_row = np.hstack([resized_images[0], resized_images[1]])
                bottom_row = np.hstack([resized_images[2], resized_images[3]])
                merged = np.vstack([top_row, bottom_row])

        # Use the first image's number in the group for naming
        first_num = int(group[0].split('_')[1].split('.')[0])
        cv2.imwrite(os.path.join(output_dir, f'merged_{first_num:03d}.png'), merged)
        logger.info(f"Merged group {i}/{total_groups} ({len(group)} images)")
    
    logger.info(f"Image merging completed: {total_groups} merged images saved")

def generate_lecture_from_images_openai(client, image_dir, prompt, faiss_textbook_indexer=None, context_size=2, model_name="gpt-4o", max_tokens=500, multiagent = False):
    """
    Generate a complete lecture by analyzing images in sequence, maintaining context.
    
    :param client: OpenAI API client
    :param image_dir: Directory containing the merged images
    :param prompt: The base prompt to use for image analysis
    :param faiss_textbook_indexer: FAISS Textbook indexer object (short term, not persisted) # FIXME: need to update to PGVector indexer
    :param context_size: Number of previous slides to consider for context
    :param model_name: The name of the model to use
    :param max_tokens: The maximum number of tokens to generate
    :param multiagent: Whether to use multi-agent refinement
    :return: A list of content for each slide, and the list of image files corresponding to each slide
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    total_slides = len(image_files)
    logger.info(f"Starting lecture generation: {total_slides} slides to process")
    
    context = []
    contents = []
    pbar = tqdm(total=len(image_files))
    pbar.set_description("Generating lecture text")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        
        # Get relevant textbook content if available
        textbook_content = None
        if faiss_textbook_indexer:
            # Extract text from image for query (you might need to implement this)
            # For now, we'll use the context as query
            query = " ".join(context[-2:]) if context else "Introduction to the topic"
            textbook_content = faiss_textbook_indexer.get_relevant_content(
                query=query,
                index_name=Path(faiss_textbook_indexer.textbook_path).stem
            ) # List[str]
            logger.debug(f"Textbook content: {textbook_content}")
        
        context_prompt = f"{prompt}\n\nContext from previous slides:\n{' '.join(context)}\n\nAnalyze the current slide in the context of what has been discussed before. Remember do not repeat the same information."
            
        # Modify prompt to include textbook content if available
        enhanced_prompt = context_prompt
        if textbook_content:
            enhanced_prompt += "\n\nRelevant textbook content:\n" + "\n".join(textbook_content)
            enhanced_prompt += "\n\nPlease incorporate relevant information from the textbook in your explanation."
        logger.debug(f"Enhanced prompt: {enhanced_prompt}")
        
        
        slide_content = analyze_page_by_image_openai(
            client, 
            enhanced_prompt, 
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
    if multiagent:
        logger.info("Starting multi-agent refinement")
        contents = multiagent_generation(client, contents, model_name=model_name)
        logger.info("Multi-agent refinement completed")
    return contents, image_files

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