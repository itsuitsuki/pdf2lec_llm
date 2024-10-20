import requests

from dotenv import load_dotenv
import os

# load the environment variables
load_dotenv()

# API key from ElevenLabs
api_key = os.getenv("ELEVENLABS_API_KEY")

# Text to be converted into speech
text = "Hi, my name is Tyler and I am your new instructor. Today we talk about linear regression. "

# Insert voice ID (copy from your dashboard; we cant do different voices!)
voice_id = "raMcNf2S8wCmuaBcyI6E"

# API endpoint for ElevenLabs Text-to-Speech
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

# Header data with API key
headers = {
    "accept": "audio/mpeg",
    "xi-api-key": api_key,
    "Content-Type": "application/json"
}

# JSON data with the text to be converted
data = {
    "text": text,
    "voice_settings": {
        "stability": 0.75,            # Stability of the voice (adjustable)
        "similarity_boost": 0.75      # Similarity to real voices (adjustable)
    }
}

# Send request to ElevenLabs API
response = requests.post(url, json=data, headers=headers)

# If the request was successful, save the audio file
if response.status_code == 200:
    with open("output_TylerInstructor.mp3", "wb") as audio_file:
        audio_file.write(response.content)
        print("Audio successfully saved!")
else:
    print(f"Error: {response.status_code}, {response.text}")
