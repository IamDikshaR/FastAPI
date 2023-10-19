import os
import whisper
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
from uvicorn import run

# Set your OpenAI API key
model = whisper.load_model("tiny")

# Define the audio transcription function
def transcribe_audio(audio_url):

    # Download the audio file from the provided URL
    with httpx.Client() as client:
        response = client.get(audio_url)
        audio_data = response.content

    # Transcribe audio using the model
    result = model.transcribe("audio.mp3")
    transcript = result["text"]

    return transcript

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(audio_url: str):
    try:
        # Transcribe audio from the provided URL
        transcript = transcribe_audio(audio_url)

        return JSONResponse(content={"transcript": transcript}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     run(app, host="0.0.0.0", port=port)
