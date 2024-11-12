import os
os.environ["TQDM_DISABLE"] = "1"
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from src.lec_gen_args import LecGenerateArgs
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

# create a Redis client and FastAPI app
redis_client = None
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

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
    
    # TODO: Upon completion of prompts, delete this warning
    if lec_args.complexity != 2:
        logger.warning("Now we have only implemented complexity 2. The complexity parameter will be ignored.")
        lec_args.complexity = 2
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
    redis_client.flushall()
    return {"message": f"Redis persistence cleared. {cleared_count} tasks were removed."}

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
    return {"message": f"All failed tasks cleared. {cleared_count} tasks were removed."}

def set_pending_to_failed():
    for key in redis_client.keys():
        task_data = redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") == "pending":
            logging.error(f"Task {key} is pending, set to failed.")
            redis_client.set(key, json.dumps({"status": "failed", "error": "Server shut down when the generation is on its way."}))
            
# delete uncompleted tasks
@app.delete("/api/v1/clear_uncompleted_tasks")
async def clear_uncompleted_tasks():
    for key in redis_client.keys():
        task_data = redis_client.get(key)
        task_info = json.loads(task_data)
        if task_info.get("status") != "completed":
            redis_client.delete(key)
    return {"message": "All uncompleted tasks cleared."}

def clear_uncompleted_data_sync():
    # delete uncompleted data/ folders & metadata
    # find all metadata filenames in metadata/
    # if the metadata does not have a corresponding data/ subfolder, delete the metadata
    # if the subfolder/generated_audios does not contain combined.mp3, delete the subfolder and metadata
    executor.shutdown(wait=False)
    executor.shutdown(wait=False, cancel_futures=True)
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
    executor.shutdown(wait=False)
    executor.shutdown(wait=False, cancel_futures=True)
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
    
    executor = ThreadPoolExecutor(max_workers=args.n_workers)
    redis_client = redis.Redis(host='localhost', port=args.redis_port, db=0)
    # delete all uncompleted data
    clear_uncompleted_data_sync()

    atexit.register(clear_uncompleted_data_sync)
    atexit.register(set_pending_to_failed)
    atexit.register(lambda: redis_client.close())
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
