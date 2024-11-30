import os
os.environ["TQDM_DISABLE"] = "1"
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
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
from openai import OpenAI
from models.user import UserCreate, UserLogin, Token
from utils.auth import get_password_hash, verify_password, create_access_token, get_current_user
import aiofiles
from fastapi.responses import FileResponse
from fastapi import Request
from jose import JWTError, jwt
from utils.auth import SECRET_KEY, ALGORITHM
from starlette.responses import Response
from starlette.types import Scope, Receive, Send

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

class AuthenticatedStaticFiles(StaticFiles):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await super().__call__(scope, receive, send)
            
        # 获取请求路径
        path = scope.get("path", "")
        
        # 获取认证头
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        
        if not auth_header or not auth_header.startswith("Bearer "):
            response = Response(
                status_code=401,
                content="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
            return await response(scope, receive, send)
            
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            
            if not email:
                response = Response(status_code=401, content="Invalid token")
                return await response(scope, receive, send)
            
            # 从路径中提取 PDF ID
            path_parts = path.split("/")
            if len(path_parts) > 2:
                pdf_id = path_parts[2]  # 获取路径中的 PDF ID
                
                # 检查用户是否有权限访问该 PDF
                user = users_db.get(email)
                if not user or pdf_id not in user.get('accessible_pdfs', []):
                    response = Response(status_code=403, content="Access denied")
                    return await response(scope, receive, send)
                    
            return await super().__call__(scope, receive, send)
            
        except JWTError:
            response = Response(status_code=401, content="Invalid token")
            return await response(scope, receive, send)

# 使用自定义的认证静态文件中间件
app.mount("/data", AuthenticatedStaticFiles(directory="data"), name="data")

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
        logger = logging.getLogger("uvicorn")
        logger.info(f"Starting new task with ID: {task_id}")
        
        # 如果使用教科书，确保从metadata中获取正确的文件名
        if lec_args.use_rag and lec_args.textbook_name:
            base_dir = f"./data/{lec_args.pdf_name}"
            with open(f"{base_dir}/metadata.json", "r") as f:
                metadata = json.load(f)
            if not metadata.get("has_textbook"):
                raise HTTPException(
                    status_code=400,
                    detail="No textbook found for this slide"
                )
            # 使用metadata中存储的教科书文件名
            lec_args.textbook_name = metadata.get("textbook_filename")
        
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
    base_dir = f"./data/{lec_args.pdf_name}"
    with open(f"{base_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
    metadata["status"] = "generating"
    with open(f"{base_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 将任务状态设置为 "generating"
    redis_client.set(task_id, json.dumps({"status": "generating"}))
    
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
    # TODO: qa_args will use use_arg and textbook_name
    logger = logging.getLogger("uvicorn")
    logger.info(f"Task {qa_args.task_id}: Question asked: {qa_args.question}")
    
    if qa_args.use_rag and qa_args.textbook_name is None:
        raise HTTPException(status_code=400, detail="textbook_name must be provided if use_rag is True.")
    
    if qa_args.use_rag and qa_args.textbook_name:
        base_dir = f"./data/{qa_args.pdf_name}"
        with open(f"{base_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        if not metadata.get("has_textbook"):
            raise HTTPException(
                status_code=400,
                detail="No textbook found for this slide"
            )
        qa_args.textbook_name = metadata.get("textbook_filename")

    

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
    base_dir = f"./data/{qa_args.pdf_name}"
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

def generate_unique_id():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = uuid.uuid4().hex[:6]  # 添加6位随机字符
    return f"{timestamp}_{random_suffix}"

@app.post("/api/v1/upload-slide")
async def upload_slide(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    logger = logging.getLogger("uvicorn")
    try:
        FileValidator.validate(file, "slide")
        sanitized_filename = FileValidator.sanitize_filename(file.filename)
        
        # 生成唯一ID
        unique_id = generate_unique_id()
        
        # 创建目录
        base_dir = f"./data/{unique_id}"
        os.makedirs(base_dir, exist_ok=True)
        
        file_path = os.path.join(base_dir, f"{sanitized_filename}")
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            
        # 创建并保存元数据
        metadata = {
            "original_filename": f"{sanitized_filename}",
            "upload_time": datetime.datetime.now().isoformat(),
            "status": "pending",
            "uploader": current_user,
            "audio_timestamps": []
        }
        
        with open(os.path.join(base_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # 将PDF ID添加到用户的可访问列表中
        user = users_db[current_user]
        if 'accessible_pdfs' not in user:
            user['accessible_pdfs'] = []
        user['accessible_pdfs'].append(unique_id)
        
        return {
            "id": unique_id,
            "filename": sanitized_filename,
            "type": "slide",
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/upload-textbook/{slide_id}")
async def upload_textbook(slide_id: str, file: UploadFile = File(...)):
    """Upload textbook PDF file"""
    logger = logging.getLogger("uvicorn")
    try:
        FileValidator.validate(file, "textbook")
        sanitized_filename = FileValidator.sanitize_filename(file.filename)
        
        # 使用传入的 slide_id 作为目录
        base_dir = f"./data/{slide_id}"
        if not os.path.exists(base_dir):
            raise HTTPException(status_code=404, detail="Slide ID not found")
            
        # 读取现有的 metadata
        with open(os.path.join(base_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        # 保存教科书文件
        pdf_path = os.path.join(base_dir, f"{sanitized_filename}")
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 更新 metadata
        metadata.update({
            "textbook_filename": sanitized_filename,
            "textbook_path": f"/data/{slide_id}/{sanitized_filename}",
            "has_textbook": True,
            "updated_at": datetime.datetime.now().isoformat()
        })
        
        # 保存更新后的 metadata
        with open(os.path.join(base_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        return JSONResponse(
            content={
                "id": slide_id,
                "filename": sanitized_filename,
                "path": f"/data/{slide_id}/{sanitized_filename}",
                "type": "textbook",
                "metadata": metadata
            },
            status_code=200
        )
    except HTTPException as e:
        logger.error(f"File validation failed: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pdfs/{type}")
async def get_pdfs(type: str, current_user: str = Depends(get_current_user)):
    """Get all PDFs of specified type"""
    logger = logging.getLogger("uvicorn")
    try:
        if type not in ["slide", "textbook"]:
            raise HTTPException(status_code=400, detail="Invalid file type")

        # 获取用户信息
        user = users_db.get(current_user)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 获取用户可访问的PDF列表
        accessible_pdfs = user.get('accessible_pdfs', [])
        
        # 获取所有PDF并过滤
        all_pdfs = get_all_pdfs(type)
        filtered_pdfs = [
            pdf for pdf in all_pdfs 
            if pdf['id'] in accessible_pdfs
        ]
        
        return filtered_pdfs
    except Exception as e:
        logger.error(f"Error getting PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/pdfs/{file_type}/{pdf_id}")
async def delete_pdf(file_type: str, pdf_id: str):
    """Delete PDF file by type and ID"""
    logger = logging.getLogger("uvicorn")
    try:
        if file_type not in ["slide", "textbook"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
            
        base_dir = f"./data/{pdf_id}"
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
            return {"message": f"{file_type} file and associated data deleted successfully"}
                
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/pdfs")
def get_all_pdfs(file_type: str) -> list:
    """
    获取指定类型的所有PDF文件
    :param file_type: 文件类型 ('slide' 或 'textbook')
    :return: PDF文件列表
    """
    pdfs = []
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        return pdfs

    for pdf_id in os.listdir(data_dir):
        pdf_path = os.path.join(data_dir, pdf_id)
        if not os.path.isdir(pdf_path):
            continue

        if file_type == "slide":
            input_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
            if input_files:
                metadata_file = os.path.join(pdf_path, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                pdfs.append({
                    "id": pdf_id,
                    "filename": input_files[0],
                    "metadata": metadata
                })

    return pdfs

@app.post("/api/v1/chatbot")
async def chatbot(request: Request):
    logger = logging.getLogger("uvicorn")
    try:
        data = await request.json()
        message = data.get('message')
        pdf_id = data.get('pdfId')  # 新增：获取 PDF ID
        
        if not message or not isinstance(message, str) or message.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"reply": "Please enter a valid message, so that I can assist you:)"}
            )
        
        # 读取讲座内容作为背景知识
        lecture_context = ""
        if pdf_id:
            try:
                base_dir = f"./data/{pdf_id}"
                with open(f"{base_dir}/lecture_content.txt", "r") as f:
                    lecture_context = f.read()
            except:
                logger.warning(f"Could not load lecture content for {pdf_id}")

        # 构建系统提示和上下文
        system_prompt = """You are a helpful teaching assistant. Use the provided lecture content to help answer student questions.
        If the question is about the lecture content, base your answer on that. If it's a general question, you can answer based on your knowledge.
        Keep answers concise and clear."""

        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # if lecture_context:
        #     messages.append({"role": "system", "content": f"Lecture content: {lecture_context}"})
        
        messages.append({"role": "user", "content": message})

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        bot_reply = response.choices[0].message.content
        logger.info(f"Bot reply: {bot_reply}")
        return {"reply": bot_reply}
        
    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"reply": "Sorry, something went wrong. Please try again later."}
        )

@app.post("/api/v1/test_image_merge")
async def test_image_merge(file: UploadFile = File(...)):
    """Test endpoint for image merging process only"""
    logger = logging.getLogger("uvicorn")
    try:
        # Validate and save the file
        FileValidator.validate(file, "slide")
        sanitized_filename = FileValidator.sanitize_filename(file.filename)
        
        # Create unique timestamp directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"test_{timestamp}_{uuid.uuid4().hex[:8]}"
        base_dir = f"./data/{unique_id}"
        
        # Create directory structure
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(f"{base_dir}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/merged_images", exist_ok=True)
        
        # Save PDF file
        pdf_path = os.path.join(base_dir, f"{sanitized_filename}")
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create metadata
        metadata = {
            "id": unique_id,
            "original_filename": sanitized_filename,
            "type": "test_merge",
            "timestamp": timestamp,
            "status": "processing"
        }
        
        with open(os.path.join(base_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Process images
        convert_pdf_to_images(pdf_path, f"{base_dir}/images")
        merge_similar_images(f"{base_dir}/images", f"{base_dir}/merged_images", 
                           similarity_threshold=0.4)
        
        metadata["status"] = "completed"
        with open(os.path.join(base_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        return JSONResponse(
            content={
                "id": unique_id,
                "path": f"/data/{unique_id}",
                "message": "Image processing completed successfully"
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error in test_image_merge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

users_db = {}

@app.post("/api/v1/register")
async def register(user: UserCreate):
    if user.email in users_db:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["id"] = str(uuid.uuid4())
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    users_db[user.email] = user_dict
    return {"message": "User created successfully"}

@app.post("/api/v1/login")
async def login(user_credentials: UserLogin):
    user = users_db.get(user_credentials.email)
    if not user or not verify_password(user_credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )
    access_token = create_access_token(data={"sub": user["email"]})
    return Token(access_token=access_token)

@app.get("/api/v1/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    user = users_db.get(current_user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "email": user["email"],
        "username": user["username"],
        "id": user["id"]
    }

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
