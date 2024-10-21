from openai import OpenAI
import sys
import os
from pathlib import Path
from pydub import AudioSegment

# Get the API key from the environment
api_key = os.environ.get('OPENAI_API_KEY')

# Set client to use the API key
client = OpenAI()

# Define paths
input_text_folder = Path(__file__).parent / "input_text"
relative_output_path = Path(__file__).parent / "test_outputs"
relative_output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

# Define the base name for the output files
base_file_name = "L15-nearest-neighbor-10-17_lecture"

# Read the input text (assuming each slide is separated by a specific marker, like "--- Slide:")
input_text = input_text_folder.read_text()

# Split the text by the slide markers
slides = input_text.split("--- Slide:")

# Keep track of all generated audio files
audio_files = []

# Loop through each slide and generate audio
for i, slide_text in enumerate(slides):
    slide_text = slide_text.strip()  # Remove leading/trailing whitespace
    
    # Skip the first item if it's empty (due to splitting at the first occurrence)
    if i == 0 and not slide_text:
        continue
    
    if slide_text:  # Process only if the slide has content
        # Remove any "--- Slide:" markers from the spoken content itself
        clean_slide_text = "\n".join([line for line in slide_text.splitlines() if not line.startswith("merged_")])

        # Define the slide audio file name
        slide_name = f"{base_file_name}.slide{i}.mp3"
        speech_file_path = relative_output_path / slide_name
        
        # Generate speech for the current slide
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=clean_slide_text  # Input cleaned text, without "--- Slide:" markers
        )
        
        # Write the response content to the respective file
        with open(speech_file_path, "wb") as f:
            f.write(response.content)
        
        print(f"Speech for Slide {i} saved at: {speech_file_path}")

        # Add the file path to the list of audio files
        audio_files.append(str(speech_file_path))

# Merge all MP3 files into one using pydub
combined = AudioSegment.empty()

for file in audio_files:
    audio_segment = AudioSegment.from_mp3(file)
    combined += audio_segment  # Concatenate audio files

# Define the final combined output file path
final_output_path = relative_output_path / f"{base_file_name}_combined.mp3"

# Export the combined audio file with higher bitrate (e.g., 192 kbps)
combined.export(final_output_path, format="mp3", bitrate="192k")

print(f"All slides have been merged into one file: {final_output_path}")
