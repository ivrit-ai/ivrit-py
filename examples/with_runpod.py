"""
Example: Using ivrit with RunPod for audio transcription
See https://www.youtube.com/watch?v=IkqArVv_Uts for how to create a RunPod endpoint.

Usage:
    wget https://github.com/ivrit-ai/asr-training/raw/refs/heads/master/examples/audio.opus
    python examples/with_runpod.py
"""
import ivrit

model = ivrit.load_model(
    engine="runpod",
    model="ivrit-ai/whisper-large-v3-turbo-ct2",
    api_key="API_KEY",
    endpoint_id="ENDPOINT_ID"
)

# Transcribe from file using new 'file' parameter
# You can also use the url parameter to transcribe from a URL
# to stream results, use stream=True
result = model.transcribe(file="audio.opus", language="he", word_timestamps=True)
print(f"Transcription: {result['text']}")
print(f"Language: {result['language']}")
