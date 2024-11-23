from utils.encode_image_to_base64 import encode_image_pil_to_base64
from openai import OpenAI
from src.arg_models import QAArgs
from typing import Optional
import logging
import os
from fastapi import HTTPException
from src.faiss_textbook_indexer import FAISSTextbookIndexer
from pathlib import Path
# uvicorn logger
logger = logging.getLogger("uvicorn")

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
    
    # FAISS
    # FIXME: PDF_NAME IS NOT IN QA_ARGS, NOT WORKING
    pdf_name = qa_args.pdf_name
    base_dir = f"./data/{pdf_name}"
    if qa_args.textbook_name and qa_args.use_rag:
        textbook_path = os.path.join(base_dir, f"Input_{qa_args.textbook_name}")
        # logger.info(f"Task {qa_args.task_id}: Initializing textbook indexer")
        logger.debug(f"Task {qa_args.task_id}: Textbook path: {textbook_path}")
        if not os.path.exists(textbook_path):
            raise HTTPException(status_code=404, detail=f"Textbook not found at {textbook_path}")
        
        faiss_textbook_indexer = FAISSTextbookIndexer(textbook_path)
        index_path = Path(faiss_textbook_indexer.index_path) / Path(textbook_path).stem
        # index_path = f"{faiss_textbook_indexer.index_path}/{os.path.splitext(textbook_path)[0]}"
        logger.debug(f"Task {qa_args.task_id}: Index path: {index_path}")
        # faiss_textbook_indexer.load_index(index_path)
        if not os.path.exists(index_path):
            logger.info(f"Task {qa_args.task_id}: Creating index")
            faiss_textbook_indexer.create_index(index_path)
        # faiss_textbook_indexer.create_index(index_path) 
        if False:
            query = " ".join([qa_args.question, transcript, pdf_content["text"]]) # FIXME: Maybe just use the question?
        
        query = " ".join([qa_args.question]) # FIXME: Maybe we should use all the context?
        textbook_content = faiss_textbook_indexer.get_relevant_content(
                query=query,
                index_name=Path(faiss_textbook_indexer.textbook_path).stem
            ) # List[str]
        logger.debug(f"Task {qa_args.task_id}: Textbook content: {textbook_content}")
        if textbook_content:
            messages.append({"role": "user", "content": f"Relevant content in the textbook:\n{textbook_content}"})
 
    # Other source context completed, now ask the question
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