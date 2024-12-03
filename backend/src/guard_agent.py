from openai import OpenAI
import json
from pathlib import Path
import logging
import os
from PIL import Image
import numpy as np
import re

logger = logging.getLogger("uvicorn")

def get_representative_slides(image_dir: str) -> list[str]:
    """
    Get representative slides from the image directory.
    Returns paths of first, last, median, Q1 and Q3 slides.
    """
    # Get all image files and sort them
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not image_files:
        raise ValueError("No image files found in directory")
    
    n = len(image_files)
    # Calculate indices for Q1, median, and Q3
    indices = [
        0,  # First slide
        (n-1) // 4,  # Q1
        (n-1) // 2,  # Median
        3 * (n-1) // 4,  # Q3
        n-1  # Last slide
    ]
    
    # Remove duplicates while maintaining order
    indices = sorted(list(set(indices)))
    
    # Get the file paths
    return [os.path.join(image_dir, image_files[i]) for i in indices]

def combine_images_vertically(image_paths: list[str]) -> str:
    """
    Combine multiple images vertically into a single image.
    Returns the path to the combined image.
    """
    # Open all images
    images = [Image.open(path) for path in image_paths]
    
    # Calculate the maximum width
    max_width = max(img.width for img in images)
    
    # Resize images to have the same width while maintaining aspect ratio
    resized_images = []
    for img in images:
        aspect_ratio = img.height / img.width
        new_height = int(max_width * aspect_ratio)
        resized_images.append(img.resize((max_width, new_height)))
    
    # Calculate total height
    total_height = sum(img.height for img in resized_images)
    
    # Create new image
    combined_image = Image.new('RGB', (max_width, total_height))
    
    # Paste images
    y_offset = 0
    for img in resized_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save combined image
    output_path = "temp_combined_slides.png"
    combined_image.save(output_path)
    return output_path

def analyze_slides_with_guard_agent(
    client: OpenAI, 
    image_paths: list[str],
    model_name: str = "gpt-4o"
) -> tuple[bool, dict]:
    """
    Analyze multiple slides to validate if they are course materials.
    
    Args:
        client: OpenAI client
        image_paths: List of paths to representative slides
        model_name: Model to use for analysis
        
    Returns:
        tuple: (is_valid, metadata)
    """
    try:
        from utils.encode_image_to_base64 import encode_image_path_to_base64
        
        # Read the guard agent prompt
        with open("prompts/guard_agent.txt", "r") as f:
            prompt = f.read()
        
        # Combine images into one
        combined_image_path = combine_images_vertically(image_paths)
        base64_image = encode_image_path_to_base64(combined_image_path)
        
        # Query GPT-4o
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
            max_tokens=500
        )
        
        # Parse the response
        json_result = response.choices[0].message.content
        logger.info(f"Guard Agent validation result: {json_result}")

        # 使用更复杂的正则表达式来匹配JSON内容
        # 首先尝试匹配markdown代码块中的JSON
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', json_result)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有markdown代码块，尝试直接匹配JSON对象
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', json_result)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No valid JSON found in response")

        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Extracted JSON string: {json_str}")
            # 提供默认响应
            result = {
                "is_course_material": False,
                "confidence": 0.0,
                "major_subject": "unknown",
                "subtopic": "unknown",
                "reasoning": f"Failed to parse response: {str(e)}"
            }
        
        # 验证并补充必需字段
        required_fields = ["is_course_material", "confidence", "major_subject", "subtopic", "reasoning"]
        for field in required_fields:
            if field not in result:
                result[field] = "unknown" if field != "is_course_material" else False
                if field != "confidence":
                    result["confidence"] = 0.0
        
        return result.get("is_course_material", False), result
        
    except Exception as e:
        logger.error(f"Guard Agent validation failed: {str(e)}")
        # 返回默认的错误响应
        error_result = {
            "is_course_material": False,
            "confidence": 0.0,
            "major_subject": "unknown",
            "subtopic": "unknown",
            "reasoning": f"Error during analysis: {str(e)}"
        }
        return False, error_result
    finally:
        # Clean up temporary file
        if os.path.exists(combined_image_path):
            os.remove(combined_image_path)

def validate_course_material(client: OpenAI, image_dir: str) -> dict:
    """
    Validate if the uploaded PDF is a legitimate course material.
    
    Args:
        client: OpenAI client
        image_dir: Directory containing the slide images
        
    Returns:
        dict: Validation result containing classification and confidence
    """
    try:
        # Get representative slides
        representative_slides = get_representative_slides(image_dir)
        logger.info(f"Selected {len(representative_slides)} representative slides for validation")
        
        # Analyze slides with Guard Agent
        is_valid, metadata = analyze_slides_with_guard_agent(client, representative_slides)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Guard Agent validation failed: {str(e)}")
        raise e
