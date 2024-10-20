from openai import OpenAI
import sys
import os
from pathlib import Path

# Get the API key from the environment
api_key = os.environ.get('OPENAI_API_KEY')

# set cline to use the API key
client = OpenAI()

input_text_file = Path(__file__).parent / "input_text"

relative_path = Path(__file__).parent / "test_outputs"

speech_file_path = relative_path / "L6_Classification_slide1.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="echo",
  input=input_text_file.read_text()
)

# Write the response content to the file manually
with open(speech_file_path, "wb") as f:
    f.write(response.content)

print(f"Speech saved at: {speech_file_path}")
