from utils.encode_image_to_base64 import encode_image_pil_to_base64
from openai import OpenAI
from src.arg_models import QAArgs
from typing import Optional

def single_gen_answer(qa_args: QAArgs, pdf_content: dict, transcript: str, past_qa_context: Optional[list[dict]] = None) -> tuple[str, list[dict]]:
    """Generate an answer using GPT-4 with both text, images, and summary context. Return the answer and the updated past QA context."""
    # Initialize the OpenAI client with the provided API key
    client = OpenAI(api_key=qa_args.openai_api_key)
    
    messages = []
    
    if transcript:
        messages.append({"role": "user", "content": f"Lecture Summary:\n{transcript}"})
    
    if pdf_content["text"]:
        messages.append({"role": "user", "content": f"Text extracted from slides:\n{pdf_content['text']}"})
    
    image_contents = []
    for image in pdf_content["images"]:
        # image: JpegImageFile
        
        base64_image = encode_image_pil_to_base64(image)
        image_contents.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        })
    messages.append({"role": "user", "content": image_contents})
    
    if past_qa_context:
        messages += past_qa_context
        past_qa_context.append({"role": "user", "content": f"Question: {qa_args.question}"})
            
    else: # no context, ask a question directly
        prompt = f"Based on previous lecture content, the provided transcript of the lecture, and the images extracted from the slides, please answer the following question.\n"
        prompt += f"\nQuestion: {qa_args.question}"
        question_dict = {"role": "user", "content": prompt} # ensure first message is with detailed prompt
        messages.append(question_dict)
        past_qa_context = [question_dict]

    try:
        response = client.chat.completions.create(
            model=qa_args.qa_model,
            messages=messages,
            max_tokens=qa_args.max_tokens
        )
        answer = response.choices[0].message.content

        if answer.startswith("Answer:"):
            answer = answer[7:].strip() 
        # add to messages, role: assistant
        past_qa_context.append({"role": "assistant", "content": answer})
        return answer, past_qa_context
        
    except Exception as e:
        raise e