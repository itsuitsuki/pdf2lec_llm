import os
os.environ["TQDM_DISABLE"] = "1"
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from src.arg_models import LecGenerateArgs, PDFSplitMergeArgs
from src.pipeline import pdf2lec
import uvicorn
import argparse
import uuid
import redis
import json
import atexit
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import partial
import shutil
from src.logger import CustomLogger
import logging
from utils.pdf_manipulation import extract_elements_from_pdf
from src.arg_models import QAArgs
from src.qa import single_gen_answer
import datetime
from src.pdf2text import convert_pdf_to_images, merge_similar_images
from typing import List
from mimetypes import guess_type
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# create a Redis client and FastAPI app
redis_client = None
qa_redis_client = None
app = FastAPI()
n_workers = 4
executor = ThreadPoolExecutor(max_workers=n_workers - n_workers // 2)
qa_executor = ThreadPoolExecutor(max_workers=n_workers // 2)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory="data"), name="data")

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "errors": exc.errors()}
    )

def configure_logging(debug_mode: bool):
    CustomLogger.setup_logger(debug_mode)

# Synchronous function to generate content
def generate_content_sync(task_id: str, lec_args: LecGenerateArgs):
    try:
        # whether to enable debug logging
        # configure_logging(lec_args.debug_mode)
        # logger = CustomLogger.get_logger(__name__)
        logger = logging.getLogger("uvicorn")
        logger.info(f"Starting new task with ID: {task_id}")
        
        # main logic to generate content
        metadata = pdf2lec(lec_args, task_id)
        # update the task status database
        redis_client.set(task_id, json.dumps({"status": "completed", "metadata": metadata}))
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        # upon failure, update the task status database
        redis_client.set(task_id, json.dumps({"status": "failed", "error": str(e)}))
        
        
# POST request interface, start the task
# POST /api/v1/lec_generate
@app.post("/api/v1/lec_generate")
async def lec_generate(lec_args: LecGenerateArgs):
    # configure logging
    # configure_logging(lec_args.debug_mode)
    # logger = CustomLogger.get_logger(__name__)
    logger = logging.getLogger("uvicorn")
    # logger.info(f"Logger setup with debug mode")
    if lec_args.debug_mode:
        os.environ["TQDM_DISABLE"] = "0"
    else:
        os.environ["TQDM_DISABLE"] = "1"
    
    if lec_args.use_rag and lec_args.textbook_name is None:
        raise HTTPException(status_code=400, detail="textbook_name must be provided if use_rag is True.")
    
    # Unique task ID
    task_id = str(uuid.uuid4())
    logger.debug(f"Task {task_id}: lec_args: {lec_args}")
    # 将任务状态设置为 "pending"
    redis_client.set(task_id, json.dumps({"status": "pending"}))
    
    # 添加后台任务
    # background_tasks.add_task(generate_content, task_id, lec_args)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, partial(generate_content_sync, task_id, lec_args))
    # see how many tasks are pending from task status database


    
    os.environ["TQDM_DISABLE"] = "1"
    # return the task ID and initial status: pending
    return {"task_id": task_id, "status": "pending"}

# GET request interface, get the task status
# GET /api/v1/task_status/{task_id}
@app.get("/api/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    task_data = redis_client.get(task_id)
    
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 解析任务数据
    task_info = json.loads(task_data)
    
    # 如果任务已完成，返回 metadata
    if task_info.get("status") == "completed":
        return {"task_id": task_id, "status": "completed", "metadata": task_info.get("metadata")}
    elif task_info.get("status") == "failed":
        return {"task_id": task_id, "status": "failed", "error": task_info.get("error")}
    else:
        # otherwise, return the task status only
        return {"task_id": task_id, "status": task_info.get("status")}

# GET statistics of tasks
# GET /api/v1/all_task_stats
@app.get("/api/v1/all_task_stats")
async def get_task_stats():
    all_pending = sum([1 for key in redis_client.keys() if json.loads(redis_client.get(key)).get("status") == "pending"])
    still_waiting = executor._work_queue.qsize() # still waiting in the executor
    logger = logging.getLogger("uvicorn")
    logger.info(f"{all_pending} tasks pending at all; ")
    logger.info(f"{still_waiting} tasks still waiting in the executor; ")
    logger.info(f"{all_pending - still_waiting} tasks are in progress.")
    
    complete_count = sum([1 for key in redis_client.keys() if json.loads(redis_client.get(key)).get("status") == "completed"])
    failed_count = sum([1 for key in redis_client.keys() if json.loads(redis_client.get(key)).get("status") == "failed"])
    # logger.debug
    logger.debug(f"{complete_count} tasks completed; ")
    logger.debug(f"{failed_count} tasks failed.")
    return {"all_pending": all_pending, "still_waiting": still_waiting, "in_progress": all_pending - still_waiting, "completed": complete_count, "failed": failed_count}
    # return {"all_pending": all_pending, "still_waiting": still_waiting, "in_progress": all_pending - still_waiting}
# DELETE /api/v1/clear_all_tasks
@app.delete("/api/v1/clear_all_tasks")
async def clear_redis_persistence():
    cleared_count = len(redis_client.keys())
    qa_cleared_count = len(qa_redis_client.keys())
    redis_client.flushall() # flush all databases
    return {"message": f"Redis persistence cleared. {cleared_count} tasks were removed. {qa_cleared_count} QA subtasks were removed."}

# delete all failed tasks
@app.delete("/api/v1/clear_failed_tasks")
async def clear_failed_tasks():
    cleared_count = 0
    for key in redis_client.keys():
        task_data = redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") == "failed":
            cleared_count += 1
            redis_client.delete(key)
    qa_cleared_count = 0
    for key in qa_redis_client.keys():
        task_data = qa_redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") == "failed":
            qa_cleared_count += 1
            qa_redis_client.delete(key)
    return {"message": f"All failed tasks cleared. {cleared_count} tasks were removed. {qa_cleared_count} QA subtasks were removed."}

def main_redis_set_panding_to_failed():
    for key in redis_client.keys():
        task_data = redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") == "pending":
            # logging.info(f"Task {key} is pending, set to failed.")
            redis_client.set(key, json.dumps({"status": "failed", "error": "Server shut down when the generation is on its way."}))
            
# delete uncompleted tasks
@app.delete("/api/v1/clear_uncompleted_tasks")
async def clear_uncompleted_tasks():
    cleared_count = 0
    for key in redis_client.keys():
        task_data = redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") != "completed":
            cleared_count += 1
            redis_client.delete(key)
    qa_cleared_count = 0
    for key in qa_redis_client.keys():
        task_data = qa_redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") != "completed":
            qa_cleared_count += 1
            qa_redis_client.delete(key)
    return {"message": f"All uncompleted tasks cleared. {cleared_count} tasks were removed. {qa_cleared_count} QA subtasks were removed."}

def clear_uncompleted_data_sync():
    # delete uncompleted data/ folders & metadata
    # find all metadata filenames in metadata/
    # if the metadata does not have a corresponding data/ subfolder, delete the metadata
    # if the subfolder/generated_audios does not contain combined.mp3, delete the subfolder and metadata
    global executor
    executor.shutdown(wait=False)
    executor.shutdown(wait=False, cancel_futures=True)
    executor = ThreadPoolExecutor(max_workers=n_workers)
    metadata_dir = "metadata/"
    data_dir = "data/"
    for metadata_filename in os.listdir(metadata_dir):
        if not os.path.isfile(os.path.join(metadata_dir, metadata_filename)): # when it's a directory, skip
            continue
        timestamp = metadata_filename.split(".")[0]
        data_folder = os.path.join(data_dir, timestamp)
        if not os.path.exists(data_folder):
            os.remove(os.path.join(metadata_dir, metadata_filename))
            continue
        # delete data/timestamp if generated_audios/ does not contain combined.mp3
        generated_audios_dir = os.path.join(data_folder, "generated_audios")
        if not os.path.exists(os.path.join(generated_audios_dir, "combined.mp3")):
            shutil.rmtree(data_folder)
            os.remove(os.path.join(metadata_dir, metadata_filename))

@app.delete("/api/v1/clear_uncompleted_data")
async def clear_uncompleted_data():
    clear_uncompleted_data_sync()
    return {"message": "All uncompleted data cleared."}

def clear_all_data_sync():
    # remove everything in metadata/ and data/
    # we end all executing tasks
    global executor
    executor.shutdown(wait=False)
    executor.shutdown(wait=False, cancel_futures=True)
    executor = ThreadPoolExecutor(max_workers=n_workers)
    metadata_dir = "metadata/"
    data_dir = "data/"
    for metadata_filename in os.listdir(metadata_dir):
        os.remove(os.path.join(metadata_dir, metadata_filename))
    for data_folder in os.listdir(data_dir):
        shutil.rmtree(os.path.join(data_dir, data_folder)) 
    
    
@app.delete("/api/v1/clear_all_data")
async def clear_all_data():
    clear_all_data_sync()
    return {"message": "All data cleared."}

def qa_sync(qa_args: QAArgs, qa_task_id: str):
    logger = logging.getLogger("uvicorn")
    logger.info(f"Task {qa_args.task_id}: Question asked: {qa_args.question}")

    task_data = redis_client.get(qa_args.task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_info = json.loads(task_data)
    metadata = task_info.get("metadata")
    pdf_path = metadata.get("pdf_src_path")
    timestamp = metadata.get("timestamp")
    transcript_path = os.path.join("data", timestamp, "generated_texts", "lecture_with_summary.txt")

    if not os.path.exists(transcript_path):
        raise HTTPException(status_code=404, detail="Summary file not found")
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    logger.info(f"Task {qa_args.task_id}: Transcript loaded.")
    pdf_content = extract_elements_from_pdf(pdf_path)
    logger.info(f"Task {qa_args.task_id}: PDF content extracted.")

    # qa_context is a list of dict of {"role": "user" or "assistant", "content": str of question or answer}
    qa_context_path = os.path.join("data", timestamp, "qa_context.json")
    if os.path.exists(qa_context_path):
        with open(qa_context_path, 'r') as f:
            qa_context = json.load(f)
    else:
        qa_context = None
    logger.info(f"Task {qa_args.task_id}: QA context loaded.")
    try:
        answer, qa_context_updated = single_gen_answer(qa_args, pdf_content, transcript, qa_context)
        # qa_context_updated: list of dict of {"role": "user" or "assistant", "content": str of question or answer}
        # save the updated qa_context to data/timestamp/qa_context.json
        # qa_context_path = os.path.join("data", timestamp, "qa_context.json")
        with open(qa_context_path, 'w') as f:
            json.dump(qa_context_updated, f)
        logger.info(f"Task {qa_args.task_id}: QA context updated and saved.")
        
        qa_redis_client.set(qa_task_id, json.dumps({"qa_task_id": qa_task_id, "status": "completed", "question": qa_args.question, "answer": answer, "qa_args": qa_args.model_dump()}))
        logger.info(f"Task {qa_args.task_id}: QA completed.")
    except RuntimeError as e:
        logger.error(f"Question answering failed: {str(e)}")
        qa_redis_client.set(qa_task_id, json.dumps({"qa_task_id": qa_task_id, "status": "failed", "error": str(e), "question": qa_args.question, "qa_args": qa_args.model_dump()}))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ask_question")
async def ask_question(qa_args: QAArgs):
    loop = asyncio.get_running_loop()
    # if qa_args.task_id as a value in qa_redis_client is pending, don't start a new task
    for key in qa_redis_client.keys():
        task_data = json.loads(qa_redis_client.get(key))
        if task_data.get("qa_args")["task_id"] == qa_args.task_id and task_data.get("status") == "pending":
            raise HTTPException(status_code=400, detail="Task already pending.")
    subtask_id = str(uuid.uuid4())
    task_data = redis_client.get(qa_args.task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = json.loads(task_data)
    if task_info.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Task is not completed")
    loop.run_in_executor(qa_executor, partial(qa_sync, qa_args, subtask_id))
    logger = logging.getLogger("uvicorn")
    logger.info(f"Task {qa_args.task_id}: QA subtask {subtask_id} started.")
    
    # return {"task_id": qa_args.task_id, "question": qa_args.question, "answer": answer}
    qa_redis_client.set(subtask_id, json.dumps({"qa_task_id": subtask_id, "status": "pending", "question": qa_args.question, "qa_args": qa_args.model_dump()}))
    return {"qa_task_id": subtask_id, "status": "pending", "question": qa_args.question, "qa_args": qa_args.model_dump()}

@app.get("/api/v1/qa_task_status/{qa_task_id}")
async def get_qa_task_status(qa_task_id: str):
    task_data = qa_redis_client.get(qa_task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task_info = json.loads(task_data)
    if task_info.get("status") == "completed":
        return {"qa_task_id": qa_task_id, "status": "completed", "question": task_info.get("question"), "answer": task_info.get("answer")}
    elif task_info.get("status") == "failed":
        return {"qa_task_id": qa_task_id, "status": "failed", "error": task_info.get("error"), "question": task_info.get("question"), "qa_args": task_info.get("qa_args")}
    else:
        return {"qa_task_id": qa_task_id, "status": task_info.get("status"), "question": task_info.get("question"), "qa_args": task_info.get("qa_args")}

def split_merge_pdf_sync(task_id: str, args: PDFSplitMergeArgs):
    try:
        logger = logging.getLogger("uvicorn")
        logger.info(f"Starting PDF split-merge task with ID: {task_id}")
        
        # Generate timestamp for unique directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        pdf_path = f"./test/{args.pdf_name}.pdf"
        image_dir = f"./data/{timestamp}/images"
        merged_image_dir = f"./data/{timestamp}/merged_images"
        
        # Convert PDF to images
        convert_pdf_to_images(pdf_path, image_dir)
        logger.info(f"Task {task_id}: PDF converted to images")
        
        # Merge similar images
        merge_similar_images(image_dir, merged_image_dir, 
                           similarity_threshold=args.similarity_threshold)
        logger.info(f"Task {task_id}: Similar images merged")
        
        metadata = {
            "timestamp": timestamp,
            "pdf_src_path": pdf_path,
            "image_dir": image_dir,
            "merged_image_dir": merged_image_dir,
            "similarity_threshold": args.similarity_threshold
        }
        
        redis_client.set(task_id, json.dumps({"status": "completed", "metadata": metadata}))
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        redis_client.set(task_id, json.dumps({"status": "failed", "error": str(e)}))

@app.post("/api/v1/split_merge_pdf")
async def split_merge_pdf(args: PDFSplitMergeArgs):
    logger = logging.getLogger("uvicorn")
    
    if args.debug_mode:
        os.environ["TQDM_DISABLE"] = "0"
    else:
        os.environ["TQDM_DISABLE"] = "1"
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    logger.debug(f"Task {task_id}: split_merge_args: {args}")
    
    # Set initial task status
    redis_client.set(task_id, json.dumps({"status": "pending"}))
    
    # Run the task in background
    loop = asyncio.get_running_loop()
    loop.run_in_executor(executor, partial(split_merge_pdf_sync, task_id, args))
    
    return {"task_id": task_id, "status": "pending"}

class FileValidator:
    MAX_FILE_SIZE = 30 * 1024 * 1024  
    ALLOWED_TYPES = ["application/pdf"]
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        # Remove potentially problematic characters, keeping only alphanumeric, dash, underscore
        import re
        # Get the name and extension separately
        name, ext = os.path.splitext(filename)
        # Replace dots with empty string in the name part only
        clean_name = name.replace('.', '')
        # Return the cleaned name with the original extension
        return f"{clean_name}{ext}"
    
    @classmethod
    def validate(cls, file: UploadFile, file_type: str) -> bool:
        # Check file type
        content_type = guess_type(file.filename)[0]
        if content_type not in cls.ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Must be one of: {cls.ALLOWED_TYPES}"
            )
        
        # Check file size
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        
        # Allow larger file size for textbooks
        max_size = cls.MAX_FILE_SIZE * 3 if file_type == "textbook" else cls.MAX_FILE_SIZE
        
        if size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size/1024/1024}MB"
            )
        
        return True

@app.post("/api/v1/upload-slide")
async def upload_slide(file: UploadFile = File(...)):
    """Upload slide PDF file"""
    logger = logging.getLogger("uvicorn")
    try:
        FileValidator.validate(file, "slide")
        os.makedirs("./data/user_uploaded_slide_pdf", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        sanitized_filename = FileValidator.sanitize_filename(file.filename)
        filename = f"{timestamp}_{sanitized_filename}"
        file_path = os.path.join("./data/user_uploaded_slide_pdf", filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Return a JSONResponse with proper headers
        return JSONResponse(
            content={
                "id": timestamp,
                "filename": sanitized_filename,  # Return original filename for display
                "path": file_path,
                "type": "slide",
                "message": "File uploaded successfully"
            },
            status_code=200
        )
    except HTTPException as e:
        logger.error(f"File validation failed: {str(e)}")
        return JSONResponse(
            content={"detail": str(e.detail)},
            status_code=e.status_code
        )
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return JSONResponse(
            content={"detail": str(e)},
            status_code=500
        )

@app.post("/api/v1/upload-textbook")
async def upload_textbook(file: UploadFile = File(...)):
    """Upload textbook PDF file"""
    logger = logging.getLogger("uvicorn")
    try:
        FileValidator.validate(file, "textbook")
        os.makedirs("./data/user_uploaded_textbook_pdf", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        sanitized_filename = FileValidator.sanitize_filename(file.filename)
        filename = f"{timestamp}_{sanitized_filename}"
        file_path = os.path.join("./data/user_uploaded_textbook_pdf", filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "id": timestamp,
            "filename": filename,
            "path": file_path,
            "type": "textbook"
        }
    except HTTPException as e:
        logger.error(f"File validation failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pdfs/{file_type}")
async def get_pdfs(file_type: str) -> List[dict]:
    """Get list of PDFs by type"""
    logger = logging.getLogger("uvicorn")
    try:
        if file_type not in ["slide", "textbook"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
            
        dir_path = f"./data/user_uploaded_{file_type}_pdf"
        logger.debug(f"Searching for PDFs in directory: {dir_path}")
        
        if not os.path.exists(dir_path):
            logger.debug(f"Directory {dir_path} does not exist")
            return []
            
        files = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".pdf"):
                timestamp = filename.split("_")[0]
                file_path = os.path.join(dir_path, filename)
                relative_path = os.path.relpath(file_path, ".")
                files.append({
                    "id": timestamp,
                    "filename": filename,
                    "path": f"/data/{relative_path}",  # 修改为正确的静态文件路径
                    "type": file_type
                })
                logger.debug(f"Found PDF file: {filename}, path: {relative_path}")
        
        return JSONResponse(
            content=files,
            headers={
                "Access-Control-Allow-Origin": "http://127.0.0.1:5173",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    except Exception as e:
        logger.error(f"Error getting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/pdfs/{file_type}/{pdf_id}")
async def delete_pdf(file_type: str, pdf_id: str):
    """Delete PDF file by type and ID"""
    logger = logging.getLogger("uvicorn")
    try:
        if file_type not in ["slide", "textbook"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Must be 'slide' or 'textbook'")
            
        dir_path = f"./data/user_uploaded_{file_type}_pdf"
        if not os.path.exists(dir_path):
            raise HTTPException(status_code=404, detail="Directory not found")
            
        for filename in os.listdir(dir_path):
            if filename.startswith(pdf_id):
                os.remove(os.path.join(dir_path, filename))
                return {"message": f"{file_type} file deleted successfully"}
                
        raise HTTPException(status_code=404, detail="File not found")
    except HTTPException as e:
        logger.error(f"File deletion failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pdfs")
async def get_all_pdfs() -> dict:
    """Get all PDFs grouped by type"""
    logger = logging.getLogger("uvicorn")
    try:
        slides = await get_pdfs("slide")
        textbooks = await get_pdfs("textbook")
        
        return {
            "slides": slides,
            "textbooks": textbooks
        }
    except Exception as e:
        logger.error(f"File retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="The port to run the server.")
    parser.add_argument("--redis_port", type=int, default=6379, help="The port of the Redis server, should be referenced from the config.")
    parser.add_argument("--n_workers", type=int, default=4, help="The number of thread workers to use.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging", default=False)
    parser.add_argument("--custom_logger", action="store_true", help="Use custom logger", default=False)
    
    args = parser.parse_args()
    
    # configure logging
    if args.custom_logger:
        configure_logging(args.debug)
    if args.debug:
        os.environ["TQDM_DISABLE"] = "0"
    else:
        os.environ["TQDM_DISABLE"] = "1"

    n_workers = args.n_workers
    executor = ThreadPoolExecutor(max_workers=n_workers)
    redis_client = redis.Redis(host='localhost', port=args.redis_port, db=0)
    qa_redis_client = redis.Redis(host='localhost', port=args.redis_port, db=1) # db=1 for QA
    # delete all uncompleted data
    clear_uncompleted_data_sync()

    # register atexit functions
    # register flushall for qa_redis_client
    # only flush db=1
    atexit.register(lambda: qa_redis_client.flushdb())
    atexit.register(clear_uncompleted_data_sync)
    atexit.register(main_redis_set_panding_to_failed)
    atexit.register(lambda: redis_client.close())
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
