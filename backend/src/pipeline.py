from openai import OpenAI
from pathlib import Path
from src.pdf2text import convert_pdf_to_images, merge_similar_images, generate_lecture_from_images_openai, digest_lecture_openai, analyze_image_with_agent
from src.tts import generate_audio_files_openai
from prompts.slide_prompts import (
    get_each_slide_prompt, 
    get_summarizing_prompt, 
    get_introduction_prompt,
    get_slide_parsing_prompt
)
from pydub import AudioSegment
import datetime
import json
import shutil
import logging
from src.faiss_textbook_indexer import FAISSTextbookIndexer
import traceback
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from src.arg_models import LecGenerateArgs
# from src.logger import CustomLogger
from src.guard_agent import validate_course_material
from tqdm import tqdm

def deep_merge_dict(dict1: dict, dict2: dict) -> dict:
    """
    recursively merge two dictionaries
    """
    merged = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in merged 
            and isinstance(merged[key], dict) 
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
            
    return merged

def pdf2lec(_args: LecGenerateArgs, task_id):
    is_successful = False
    logger = logging.getLogger("uvicorn") 
    logger.setLevel(logging.DEBUG if _args.debug_mode else logging.INFO)
    logger.info(f"Task {task_id} started.")
    try:
        SIMILARITY_THRESHOLD_TO_MERGE = _args.similarity_threshold
        TEXT_GENERATING_CONTEXT_SIZE = _args.text_generating_context_size
        MAX_TOKENS = _args.max_tokens

        TEST_PDF_NAME = _args.pdf_name.replace('.', '')
        TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        PDF_PATH = f"./data/user_uploaded_slide_pdf/{TEST_PDF_NAME}.pdf"

        PAGE_MODEL = "gpt-4o"
        DIGEST_MODEL = "gpt-4o-mini"
        TTS_MODEL = "tts-1"
        TTS_VOICE = "alloy"

        # if True, regenerate the lecture content and summary regardless of whether the files already exist
        # not used
        DO_REGENERATE = False

        METADATA = {
            "title": TEST_PDF_NAME,
            "author": "default",
            "timestamp": TIMESTAMP,
            "pdf_src_path": PDF_PATH,
            "similarity_threshold": SIMILARITY_THRESHOLD_TO_MERGE,
            "text_generating_context_size": TEXT_GENERATING_CONTEXT_SIZE,
            "max_tokens": MAX_TOKENS,
            "page_model": PAGE_MODEL,
            "digest_model": DIGEST_MODEL,
            "tts_model": TTS_MODEL,
            "tts_voice": TTS_VOICE,
            "use_rag": _args.use_rag,
            "textbook_name": _args.textbook_name,
            "multiagent": _args.multiagent,
            "audio_timestamps": []
        }
        client = OpenAI(api_key=_args.openai_api_key)

        logger.debug(f"Task {task_id}: Starting with configuration: {json.dumps(METADATA, indent=2)}")
        

        logger.info(f"Task {task_id}: Lecture Text Generation")
        pdf_id = _args.pdf_name
        base_dir = f"./data/{pdf_id}"
        
        with open(f"{base_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        
        original_filename = metadata.get('original_filename')
        PDF_PATH = f"{base_dir}/{original_filename}"

        generated_lecture_dir = f"{base_dir}/generated_texts"
        audio_dir = f"{base_dir}/generated_audios"
        image_dir = f"{base_dir}/images"
        merged_image_dir = f"{base_dir}/merged_images"
        metadata_file = f"{base_dir}/metadata.json"
        
        # 读取原始metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # 使用深度合并更新METADATA
        METADATA = deep_merge_dict(METADATA, metadata)
        METADATA = deep_merge_dict(METADATA, {
            "timestamp": TIMESTAMP,
            "similarity_threshold": SIMILARITY_THRESHOLD_TO_MERGE,
        })
        
        # 保存更新后的metadata
        with open(f"{base_dir}/metadata.json", "w") as f:
            json.dump(METADATA, f, ensure_ascii=False, indent=4)

        is_text_generated = Path(generated_lecture_dir).exists() and Path(
            f"{generated_lecture_dir}/lecture/summary.txt").exists()
        if is_text_generated and not DO_REGENERATE:
            # rebuild content_list from the saved files

            content_list = []
            for file in Path(f"{generated_lecture_dir}/lecture/").iterdir():
                with open(file, 'r', encoding='utf-8') as f:
                    content_list.append(f.read())  # including the summary
            # print("The lecture content and summary have been loaded from the saved files.")
            logger.info(f"Task {task_id}: The lecture content and summary have been loaded from the saved files.")
        else:

            Path(generated_lecture_dir).mkdir(parents=True, exist_ok=True)
            Path(f"{generated_lecture_dir}/lecture").mkdir(parents=True, exist_ok=True)

            # Convert PDF to images
            convert_pdf_to_images(PDF_PATH, image_dir)
            
            # Validate course material using Guard Agent
            logger.info(f"Task {task_id}: Validating course material")
            validation_result = validate_course_material(client, image_dir)
            
            # Update metadata with validation result
            METADATA.update({
                "validation": validation_result
            })
            
            # If not a valid course material, set status and return metadata
            if not validation_result.get("is_course_material", False):
                METADATA["status"] = "non_lecture_pdf_error"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(METADATA, f, ensure_ascii=False, indent=4)
                return METADATA  # Return metadata instead of raising exception
            
            # Continue with normal processing if it is a valid course material
            merge_similar_images(image_dir, merged_image_dir,
                                similarity_threshold=SIMILARITY_THRESHOLD_TO_MERGE)

            # Initialize textbook indexer if textbook path is provided
            faiss_textbook_indexer = None
            
            
            if _args.textbook_name and _args.use_rag:
                # 使用同一目录下的教科书文件
                textbook_path = f"{base_dir}/{_args.textbook_name}"
                logger.info(f"Task {task_id}: Initializing textbook indexer")
                logger.debug(f"Task {task_id}: Textbook path: {textbook_path}")
                
                if not os.path.exists(textbook_path):
                    raise FileNotFoundError(f"Textbook file not found at {textbook_path}")
                    
                faiss_textbook_indexer = FAISSTextbookIndexer(textbook_path)
                # Create index if it doesn't exist
                index_path = Path(faiss_textbook_indexer.index_path) / Path(textbook_path).stem
                if not index_path.exists():
                    logger.info(f"Task {task_id}: Creating textbook index")
                    faiss_textbook_indexer.create_index()

            # Get both prompts
            slide_parsing_prompt = get_slide_parsing_prompt()

            # Create a directory for slide analysis results
            slide_analysis_dir = f"{base_dir}/slide_analysis"
            Path(slide_analysis_dir).mkdir(parents=True, exist_ok=True)

            content_list, image_files, slide_analyses = generate_lecture_from_images_openai(
                client,
                merged_image_dir,
                parsing_prompt=slide_parsing_prompt,
                faiss_textbook_indexer=faiss_textbook_indexer,
                context_size=TEXT_GENERATING_CONTEXT_SIZE,
                model_name=PAGE_MODEL,
                max_tokens=MAX_TOKENS,
                multiagent=_args.multiagent,
            )

            # Save slide analyses
            analysis_output = {
                "timestamp": TIMESTAMP,
                "total_slides": len(slide_analyses),
                "analyses": []
            }

            for i, (analysis, image_file) in enumerate(zip(slide_analyses, image_files)):
                analysis_entry = {
                    "slide_number": i + 1,
                    "image_file": image_file,
                    "analysis": analysis
                }
                analysis_output["analyses"].append(analysis_entry)

            # Save to JSON file
            analysis_file = f"{slide_analysis_dir}/slide_analyses.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_output, f, ensure_ascii=False, indent=4)

            logger.info(f"Task {task_id}: Slide analyses saved to {analysis_file}")

            # Update metadata with analysis information
            METADATA.update({
                "slide_analysis": {
                    "file_path": analysis_file,
                    "total_slides": len(slide_analyses),
                    "timestamp": TIMESTAMP
                }
            })

            # save each content into a separate file
            for i, content in enumerate(content_list):
                three_digit_number = str(i).zfill(3)
                with open(f"{generated_lecture_dir}/lecture/page_{three_digit_number}.txt", 'w', encoding='utf-8') as f:
                    f.write(content)
            # print(
            #     f"Lecture content saved to {generated_lecture_dir}/lecture/")
            logger.info(f"Task {task_id}: Lecture content saved to {generated_lecture_dir}/lecture/")

            # combine all the content into one string
            complete_lecture = ""
            for i, content in enumerate(content_list):
                complete_lecture += f"----- Slide {i+1} -----\n"
                complete_lecture += content
                complete_lecture += "\n\n"

            # Introduction generation
            logger.debug(f"Task {task_id}: Using introduction prompt:")
            logger.debug(get_introduction_prompt())

            introduction = digest_lecture_openai(
                client, complete_lecture, get_introduction_prompt(), model_name=DIGEST_MODEL)

            logger.debug(f"Task {task_id}: Generated introduction:")
            logger.debug(introduction)

            # Summary generation
            logger.debug(f"Task {task_id}: Using summary prompt:")
            logger.debug(get_summarizing_prompt())

            summary = digest_lecture_openai(
                client, complete_lecture, get_summarizing_prompt(), model_name=DIGEST_MODEL)

            logger.debug(f"Task {task_id}: Generated summary:")
            logger.debug(summary)

            # add the introduction to the beginning of the lecture
            content_list = [introduction] + content_list
            with open(f"{generated_lecture_dir}/lecture/introduction.txt", 'w', encoding='utf-8') as f:
                f.write(introduction)
                
            # Summarization
            summary = digest_lecture_openai(
                client, complete_lecture, get_summarizing_prompt(), model_name=DIGEST_MODEL)
            content_list.append(f"Here is the summary. \n{summary}")
            with open(f"{generated_lecture_dir}/lecture/summary.txt", 'w', encoding='utf-8') as f:
                f.write(summary)
            # print(
            #     f"Summary saved to {generated_lecture_dir}/lecture/summary.txt")
            logger.info(f"Task {task_id}: Summary saved to {generated_lecture_dir}/lecture/summary.txt")
            with open(f"{generated_lecture_dir}/lecture_with_summary.txt", 'w', encoding='utf-8') as f:
                f.write("Introduction:\n\n")
                f.write(introduction)
                f.write("="*50)
                f.write("\n\n")
                f.write("Content:\n\n")
                f.write(complete_lecture)
                f.write("="*50)
                f.write("\n\n")
                f.write("Summary:\n\n")
                f.write(summary)
            # print(f"The whole lecture including the introduction, the main content, and the summary, "
            #     f"has been saved to {generated_lecture_dir}/whole_lecture.txt")
            logger.info(f"Task {task_id}: The whole lecture including the introduction, the main content, and the summary, "
                f"has been saved to {generated_lecture_dir}/whole_lecture.txt")
        # print("===== Audio Generation =====")
        logger.info(f"Task {task_id}: Audio Generation")

        is_audio_generated = Path(audio_dir).exists() and Path(
            f"{audio_dir}/summary.mp3").exists()
        if is_audio_generated and not DO_REGENERATE:
            logger.info(f"Task {task_id}: Loading existing audio files")
            audio_files = list(Path(audio_dir).iterdir())
            # delete ends with summary.mp3
            audio_files = [file for file in audio_files if not str(
                file).endswith("summary.mp3")]
            # delete ends with combined.mp3
            audio_files = [file for file in audio_files if not str(
                file).endswith("combined.mp3")]
            # delete ends with introduction.mp3
            audio_files = [file for file in audio_files if not str(
                file).endswith("introduction.mp3")]
            
            # sort the files by the page number
            audio_files.sort(key=lambda x: int(x.stem.split("_")[1]))
            # add the introduction audio file at the beginning
            audio_files.insert(0, str(Path(f"{audio_dir}/introduction.mp3")))
            # add the summary audio file at the end
            audio_files.append(str(Path(f"{audio_dir}/summary.mp3")))
        else:
            logger.info(f"Task {task_id}: Starting parallel audio generation")
            Path(audio_dir).mkdir(parents=True, exist_ok=True)
            
            # 创建线程安全的列表来存储生成的音频文件路径
            audio_files_lock = threading.Lock()
            audio_files = []
            
            def generate_single_audio(args):
                index, content = args
                try:
                    file_path = str(Path(audio_dir) / f"page_{index:03d}.mp3")
                    if index == 0:
                        file_path = str(Path(audio_dir) / "introduction.mp3")
                    elif index == len(content_list) - 1:
                        file_path = str(Path(audio_dir) / "summary.mp3")
                    
                    # 生成单个音频文件
                    response = client.audio.speech.create(
                        model=TTS_MODEL,
                        voice=TTS_VOICE,
                        input=content
                    )
                    
                    # 保存音频文件
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                    
                    # 线程安全地添加文件路径
                    with audio_files_lock:
                        audio_files.append(file_path)
                        logger.info(f"Task {task_id}: Generated audio {len(audio_files)}/{len(content_list)}")
                    
                    return file_path
                except Exception as e:
                    logger.error(f"Task {task_id}: Failed to generate audio for page {index}: {str(e)}")
                    raise e
            
            # 使用线程池并发生成音频
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 创建任务列表，包含索引和内容
                tasks = list(enumerate(content_list))
                
                # 提交所有任务并等待完成
                list(executor.map(generate_single_audio, tasks))
            
            # 按正确顺序排序音频文件
            audio_files.sort(key=lambda x: int(Path(x).stem.split("_")[1]) if "_" in Path(x).stem else -1 if "introduction" in x else 999)
            
            logger.info(f"Task {task_id}: All audio files generated successfully")

        # Merge all the audio files into one
        # print("===== Merging Audio Files =====")
        logger.info(f"Task {task_id}: Merging Audio Files")
        combined = AudioSegment.empty()
        current_position = 0
        audio_timestamps = [0]

        for file in audio_files:
            audio_segment = AudioSegment.from_mp3(file)
            combined += audio_segment
            current_position += len(audio_segment)
            audio_timestamps.append(current_position)
        
        audio_timestamps.pop()
        logger.debug(f"Task {task_id}: Audio timestamps: {audio_timestamps}")
        final_output_path = f"{audio_dir}/combined.mp3"
        combined.export(final_output_path, format="mp3", bitrate="192k")
        
        METADATA["status"] = "completed"
        METADATA["audio_timestamps"] = audio_timestamps
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(METADATA, f, ensure_ascii=False, indent=4)
        logger.info(f"Task {task_id}: All audio files have been merged into one file: {final_output_path}")
        is_successful = True
        return METADATA # dict
    except Exception as e:
        error_msg = f"Task {task_id} failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise e
    finally:
        if is_successful:
            logger.info(f"Task {task_id} finished.")
        else:
            # Only cleanup if it's not a non_lecture_pdf_error
            if METADATA.get("status") != "non_lecture_pdf_error":
                cleanup_paths = [
                    f"{base_dir}/generated_texts",
                    f"{base_dir}/generated_audios",
                    f"{base_dir}/images",
                    f"{base_dir}/merged_images"
                ]
                for cleanup_path in cleanup_paths:
                    if Path(cleanup_path).exists():
                        shutil.rmtree(cleanup_path)
                logger.error(f"Task {task_id}: Generated files have been removed due to failure or interruption.")
            

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--similarity_threshold", type=float, default=0.7, help="The similarity threshold to merge similar images.")
#     parser.add_argument("--text_generating_context_size", type=int, default=2, help="The context size for text generation, in the unit of slides.")
#     parser.add_argument("--max_tokens", type=int, default=1000, help="The maximum number of tokens for text generation.")
#     # parser.add_argument("--do_regenerate", action="store_true", help="Whether to regenerate the lecture content and summary regardless of whether the files already exist.")
#     parser.add_argument("--pdf_name", type=str, default="handout4_binary_hypothesis_testing", help="The name of the PDF file.")
#     parser.add_argument("--page_model", type=str, default="gpt-4o", help="The model name for generating the content of each page.")
#     parser.add_argument("--digest_model", type=str, default="gpt-4o-mini", help="The model name for generating the introduction and summary.")
#     parser.add_argument("--tts_model", type=str, default="tts-1", help="The model name for generating the audio.")
#     parser.add_argument("--tts_voice", type=str, default="alloy", help="The voice for generating the audio.")

#     args = parser.parse_args()
#     pdf2lec(args)