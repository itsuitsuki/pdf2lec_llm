from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def generate_audio_files_openai(client, content_list, audio_dir, model_name="tts-1", voice="alloy"):
    """
    Generate audio files for each slide in the content list and save them in the specified directory.
    
    :param client: OpenAI client instance.
    :param content_list: List of strings, where each string represents the content of a slide.
    :param audio_dir: Directory path to save the generated audio files.
    :param model_name: Name of the model to use for text-to-speech conversion.
    :param voice: Voice to use for text-to-speech conversion.
    :return: List of file paths for the generated audio files.
    """
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    for file in Path(audio_dir).iterdir():
        file.unlink()
    audio_files = []

    
    clean_slide_text_intro = content_list[0].strip()  # Remove leading/trailing whitespace
    if clean_slide_text_intro:  # Process only if the slide has content
        # Define the slide audio file name
        slide_name = "introduction.mp3"
        speech_file_path = f"{audio_dir}/{slide_name}"

        # Generate speech for the current slide
        response = client.audio.speech.create(
            model=model_name,
            voice=voice,
            input=clean_slide_text_intro  # Input cleaned text, without "--- Slide:" markers
        )

        # Write the response content to the respective file
        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        print(f"Speech for the introduction saved at: {speech_file_path}")

        # Add the file path to the list of audio files
        audio_files.append(str(speech_file_path))
    pbar = tqdm(total=len(content_list)-2, position=0, leave=True)
    pbar.set_description("Generating audio files")
    for i, slide_text in enumerate(content_list[1:-1]):  # Skip the last item, which is the summary
        clean_slide_text = slide_text.strip()  # Remove leading/trailing whitespace

        if clean_slide_text:  # Process only if the slide has content
            # Define the slide audio file name
            slide_name = f"page_{str(i + 1).zfill(3)}.mp3" # zero-padded 3-digit number like "page_001.mp3"
            speech_file_path = f"{audio_dir}/{slide_name}"

            # Generate speech for the current slide
            response = client.audio.speech.create(
                model=model_name,
                voice=voice,
                input=clean_slide_text  # Input cleaned text, without "--- Slide:" markers
            )

            # Write the response content to the respective file
            with open(speech_file_path, "wb") as f:
                f.write(response.content)

            pbar.update(1)
            pbar.set_postfix_str(f"Speech for slide {i+1} saved at: {speech_file_path}")

            # Add the file path to the list of audio files
            audio_files.append(str(speech_file_path))
            
    # summarize the lecture
    clean_slide_text_summary = content_list[-1].strip()  # Remove leading/trailing whitespace
    if clean_slide_text_summary:  # Process only if the slide has content
        # Define the slide audio file name
        slide_name = "summary.mp3"
        speech_file_path = f"{audio_dir}/{slide_name}"

        # Generate speech for the current slide
        response = client.audio.speech.create(
            model=model_name,
            voice=voice,
            input=clean_slide_text_summary  # Input cleaned text, without "--- Slide:" markers
        )

        # Write the response content to the respective file
        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        print(f"Speech for the summary saved at: {speech_file_path}")

        # Add the file path to the list of audio files
        audio_files.append(str(speech_file_path))
        
    pbar.close()
    return audio_files