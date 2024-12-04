from openai import OpenAI
from pathlib import Path
from src.pdf2text import convert_pdf_to_images, merge_similar_images, generate_lecture_from_images_openai, summarize_lecture_openai
from src.tts import generate_audio_files_openai
from prompts.slide_prompts import get_each_page_prompt, get_summarizing_prompt, get_detailed_prompt, get_brief_prompt
from pydub import AudioSegment
from tqdm import tqdm

SIMILARITY_THRESHOLD_TO_MERGE = 0.4
TEXT_GENERATING_CONTEXT_SIZE = 2
MAX_TOKENS = 1000

TEST_PDF_NAME = "L23-MDP-11-14"
PDF_PATH = f"./test/{TEST_PDF_NAME}.pdf"

PAGE_MODEL = "gpt-4o"
SUMMARIZATION_MODEL = "gpt-4o-mini"
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"
COMPLEXITY = 3 # 1=brief, 2=default, 3=detailed

# NOTE: if True, regenerate the lecture content and summary regardless of whether the files already exist
DO_REGENERATE = True
USE_MULTIAGENT = False

client = OpenAI()

print("===== Lecture Text Generation =====")
# generated_lecture_dir = f"./data/generated_texts/{TEST_PDF_NAME}"
# audio_dir = f"./data/generated_audios/{TEST_PDF_NAME}"
# image_dir = f"./data/images/{TEST_PDF_NAME}"
# merged_image_dir = f"./data/merged_images/{TEST_PDF_NAME}"
generated_lecture_dir = f"./data/{TEST_PDF_NAME}/generated_texts"
audio_dir = f"./data/{TEST_PDF_NAME}/generated_audios"
image_dir = f"./data/{TEST_PDF_NAME}/images"
merged_image_dir = f"./data/{TEST_PDF_NAME}/merged_images"

is_text_generated = Path(generated_lecture_dir).exists() and Path(
    f"{generated_lecture_dir}/lecture/summary.txt").exists()
if is_text_generated and not DO_REGENERATE:
    # rebuild content_list from the saved files

    content_list = []
    for file in Path(f"{generated_lecture_dir}/lecture/").iterdir():
        with open(file, 'r', encoding='utf-8') as f:
            content_list.append(f.read())  # including the summary
    print("The lecture content and summary have been loaded from the saved files.")
else:

    Path(generated_lecture_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{generated_lecture_dir}/lecture").mkdir(parents=True, exist_ok=True)

    convert_pdf_to_images(PDF_PATH, image_dir)
    merge_similar_images(image_dir, merged_image_dir,
                         similarity_threshold=SIMILARITY_THRESHOLD_TO_MERGE)

    content_list, image_files = generate_lecture_from_images_openai(client, merged_image_dir, get_each_page_prompt(COMPLEXITY),
                                                                     context_size=TEXT_GENERATING_CONTEXT_SIZE, model_name=PAGE_MODEL, 
                                                                     max_tokens=MAX_TOKENS, multiagent=USE_MULTIAGENT)

    # save each content into a separate file
    for i, content in enumerate(content_list):
        image_file = image_files[i]
        three_digit_number = str(i).zfill(3)
        with open(f"{generated_lecture_dir}/lecture/page_{three_digit_number}.txt", 'w', encoding='utf-8') as f:
            f.write(content)
    print(
        f"Lecture content saved to {generated_lecture_dir}/lecture/")

    # combine all the content into one string
    complete_lecture = ""
    for i, content in enumerate(content_list):
        complete_lecture += f"----- Slide {i+1} -----\n"
        complete_lecture += content
        complete_lecture += "\n\n"

    summary = summarize_lecture_openai(
        client, complete_lecture, get_summarizing_prompt(), model_name=SUMMARIZATION_MODEL)
    content_list.append(f"Here is the summary. \n{summary}")
    with open(f"{generated_lecture_dir}/lecture/summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    print(
        f"Summary saved to './data/generated_texts/{TEST_PDF_NAME}/lecture/summary.txt'")
    with open(f"{generated_lecture_dir}/lecture_with_summary.txt", 'w', encoding='utf-8') as f:
        f.write("Complete Lecture:\n\n")
        f.write(complete_lecture)
        f.write("\n\nSummary:\n\n")
        f.write(summary)
    print(f"Lecture content and summary have been saved together to {generated_lecture_dir}/lecture_with_summary.txt")

print("===== Audio Generation =====")

is_audio_generated = Path(audio_dir).exists() and Path(f"{audio_dir}/summary.mp3").exists()
if is_audio_generated and not DO_REGENERATE:
    print("The audio files have already been generated.")
    audio_files = list(Path(audio_dir).iterdir())
    # delete ends with summary.mp3
    audio_files = [file for file in audio_files if not str(file).endswith("summary.mp3")]
    # sort the files by the page number
    audio_files.sort(key=lambda x: int(x.stem.split("_")[1]))
    # add the summary audio file at the end
    audio_files.append(str(Path(f"{audio_dir}/summary.mp3")))
else:
    # delete the existing audio files
    audio_files = generate_audio_files_openai(
        client, content_list, audio_dir, model_name=TTS_MODEL, voice=TTS_VOICE)
    
# Merge all the audio files into one
print("===== Merging Audio Files =====")
combined = AudioSegment.empty()
for file in audio_files:
    audio_segment = AudioSegment.from_mp3(file)
    combined += audio_segment  # Concatenate audio files
final_output_path = f"{audio_dir}/combined.mp3"
combined.export(final_output_path, format="mp3", bitrate="192k")

print(f"All audio files have been merged into one file: {final_output_path}")
