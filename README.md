# ivrit

Python package providing wrappers around ivrit.ai's capabilities.

## Installation

```bash
pip install ivrit
```

## Usage

### Audio Transcription

The `ivrit` package provides audio transcription functionality using multiple engines.

#### Basic Usage

```python
import ivrit

# Transcribe a local audio file
model = ivrit.load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2")
result = model.transcribe(path="audio.mp3")

# With custom device
model = ivrit.load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2", device="cpu")
result = model.transcribe(path="audio.mp3")

print(result["text"])
```

#### Speaker Diarization

Enable speaker diarization to identify "who spoke when" in your audio files.
We use [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.

⚠️ Note: Since it's based on an off-the-shelf model, speaker diarization is not perfect, and is provided as a best-effort.

**Prerequisites:**
1. Accept [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) user conditions
2. Accept [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) user conditions  
3. Create access token at [hf.co/settings/tokens](https://hf.co/settings/tokens) and (optionally) set as environment variable (e.g., `HF_TOKEN`)

```python
# Transcribe with speaker diarization
model = ivrit.load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2")
result = model.transcribe(path="audio.mp3", diarize=True)

# Access speaker information
for segment in result["segments"]:
    speaker = segment.get("speaker", "Unknown")
    print(f"Speaker {speaker}: {segment['text']}")

# With additional diarization options
result = model.transcribe(
    path="audio.mp3",
    diarize=True,
    num_speakers=2,  # Specify exact number of speakers
    # min_speakers=2,  # Or specify range
    # max_speakers=4
)

# Alternative: pass token explicitly (if not set as environment variable)
result = model.transcribe(path="audio.mp3", diarize=True, use_auth_token="HF_TOKEN_GOES_HERE")
```

#### Transcribe from URL

```python
# Transcribe audio from a URL
model = ivrit.load_model(engine="faster-whisper", model="ivrit-ai/whisper-large-v3-turbo-ct2")
result = model.transcribe(url="https://example.com/audio.mp3")

print(result["text"])
```

#### Streaming Results

```python
# Get results as a stream (generator)
model = ivrit.load_model(engine="faster-whisper", model="base")
for segment in model.transcribe(path="audio.mp3", stream=True, verbose=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")

# Or use the model directly
model = ivrit.FasterWhisperModel(model="base")
for segment in model.transcribe(path="audio.mp3", stream=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")

# Access word-level timing
for segment in model.transcribe(path="audio.mp3", stream=True):
    print(f"Segment: {segment.text}")
    for word in segment.extra_data.get('words', []):
        print(f"  {word['start']:.2f}s - {word['end']:.2f}s: '{word['word']}'")

# Streaming with speaker diarization
for segment in model.transcribe(path="audio.mp3", stream=True, diarize=True):
    speakers = segment.speakers[0] if segment.speakers else "Unknown"
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: Speaker {speakers}: {segment.text}")
```

## API Reference

### `load_model()`

Load a transcription model for the specified engine and model.

#### Parameters

- **engine** (`str`): Transcription engine to use. Options: `"faster-whisper"`, `"stable-ts"`
- **model** (`str`): Model name for the selected engine
- **device** (`str`, optional): Device to use for inference. Default: `"auto"`. Options: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:0"`, etc.
- **model_path** (`str`, optional): Custom path to the model (for faster-whisper)

#### Returns

- `TranscriptionModel` object that can be used for transcription

#### Raises

- `ValueError`: If the engine is not supported
- `ImportError`: If required dependencies are not installed

### `TranscriptionModel.transcribe()`

Transcribe audio with optional speaker diarization.

#### Parameters

- **path** (`str`, optional): Path to the audio file to transcribe (mutually exclusive with url)
- **url** (`str`, optional): URL to download and transcribe (mutually exclusive with path)  
- **language** (`str`, optional): Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
- **stream** (`bool`, optional): Whether to return results as a generator (True) or full result (False). Default: `False`
- **diarize** (`bool`, optional): Whether to enable speaker diarization. Default: `False`
- **verbose** (`bool`, optional): Whether to enable verbose output. Default: `False`
- **kwargs**: Additional keyword arguments for the transcription model and diarization. For diarization options, see [pyannote.audio documentation](https://github.com/pyannote/pyannote-audio).

#### Returns

- Dictionary with transcription results (when `stream=False`) or Generator of `Segment` objects (when `stream=True`)



## Architecture

The ivrit package uses an object-oriented design with a base `TranscriptionModel` class and specific implementations for each transcription engine.

### Model Classes

- **`TranscriptionModel`**: Abstract base class for all transcription models
- **`FasterWhisperModel`**: Implementation for the Faster Whisper engine

### Usage Patterns

#### Pattern 1: Using `load_model()` (Recommended)
```python
# Step 1: Load the model
model = ivrit.load_model(engine="faster-whisper", model="base")

# Step 2: Transcribe audio
result = model.transcribe(path="audio.mp3")
```

#### Pattern 2: Direct Model Creation
```python
# Create model directly
model = ivrit.FasterWhisperModel(model="base")

# Use the model
result = model.transcribe(path="audio.mp3")
```

### Multiple Transcriptions
For multiple transcriptions, load the model once and reuse it:
```python
# Load model once
model = ivrit.load_model(engine="faster-whisper", model="base")

# Use for multiple transcriptions
result1 = model.transcribe(path="audio1.mp3")
result2 = model.transcribe(path="audio2.mp3")
result3 = model.transcribe(path="audio3.mp3")
```

## Installation

### Basic Installation
```bash
pip install ivrit
```

### With Faster Whisper Support
```bash
pip install ivrit[faster-whisper]
```

## Supported Engines

### faster-whisper
Fast and accurate speech recognition using the Faster Whisper model.

**Model Class**: `FasterWhisperModel`

**Available Models**: `base`, `large`, `small`, `medium`, `large-v2`, `large-v3`

**Features**:
- Word-level timing information
- Language detection with confidence scores
- Support for custom devices (CPU, CUDA, etc.)
- Support for custom model paths
- Streaming transcription

**Dependencies**: `faster-whisper>=1.1.1`

### stable-ts
Stable and reliable transcription using Stable-TS models.

**Status**: Not yet implemented

## Development

### Installation for Development

```bash
git clone <repository-url>
cd ivrit
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
isort .
```

## License

MIT License - see LICENSE file for details. 