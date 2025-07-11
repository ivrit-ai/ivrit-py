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
from ivrit import transcribe, load_model, FasterWhisperModel

# Transcribe a local audio file
model = load_model(engine="faster-whisper", model="base")
result = transcribe(transcription_model=model, path="audio.mp3")

# Or use the model directly
model = FasterWhisperModel(model="base")
result = model.transcribe(path="audio.mp3")

# With custom device
model = load_model(engine="faster-whisper", model="base", device="cpu")
result = model.transcribe(path="audio.mp3")

print(result["text"])
```

#### Transcribe from URL

```python
# Transcribe audio from a URL
model = load_model(engine="faster-whisper", model="base")
result = transcribe(transcription_model=model, url="https://example.com/audio.mp3")

# Or use the model directly
model = FasterWhisperModel(model="base")
result = model.transcribe(url="https://example.com/audio.mp3")

print(result["text"])
```

#### Streaming Results

```python
# Get results as a stream (generator)
model = load_model(engine="faster-whisper", model="base")
for segment in transcribe(
    transcription_model=model,
    path="audio.mp3",
    stream=True,
    verbose=True
):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")

# Or use the model directly
model = FasterWhisperModel(model="base")
for segment in model.transcribe(path="audio.mp3", stream=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")

# Access word-level timing
for segment in model.transcribe(path="audio.mp3", stream=True):
    print(f"Segment: {segment.text}")
    for word in segment.extra_data.get('words', []):
        print(f"  {word['start']:.2f}s - {word['end']:.2f}s: '{word['word']}'")
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

### `transcribe()`

Transcribe audio using ivrit.ai's transcription services.

#### Parameters

- **transcription_model** (`TranscriptionModel`): Pre-loaded TranscriptionModel object (required)
- **path** (`str`, optional): Path to the audio file to transcribe (mutually exclusive with `url`)
- **url** (`str`, optional): URL to download and transcribe (mutually exclusive with `path`)
- **stream** (`bool`, optional): Whether to return results as a generator. Default: `False`
- **verbose** (`bool`, optional): Whether to enable verbose output. Default: `False`

#### Returns

- If `stream=False`: Complete transcription result as dictionary
- If `stream=True`: Generator yielding transcription segments

#### Raises

- `ValueError`: If both `path` and `url` are provided, or neither is provided
- `FileNotFoundError`: If the specified path doesn't exist
- `Exception`: For other transcription errors

#### Example Response Format

```python
{
    "text": "Complete transcribed text",
    "segments": [
        Segment(
            text="Segment text",
            start=0.0,
            end=5.0,
            extra_data={
                "info": {
                    "language": "he",
                    "language_probability": 0.95
                },
                "words": [
                    {
                        "start": 0.0,
                        "end": 1.2,
                        "word": "Hello",
                        "probability": 0.98
                    }
                ]
            }
        )
    ],
    "language": "he",
    "engine": "faster-whisper",
    "model": "base"
}
```

## Architecture

The ivrit package uses an object-oriented design with a base `TranscriptionModel` class and specific implementations for each transcription engine.

### Model Classes

- **`TranscriptionModel`**: Abstract base class for all transcription models
- **`FasterWhisperModel`**: Implementation for the Faster Whisper engine

### Usage Patterns

#### Pattern 1: Using `load_model()` (Recommended)
```python
# Step 1: Load the model
model = load_model(engine="faster-whisper", model="base")

# Step 2: Transcribe audio
result = transcribe(transcription_model=model, path="audio.mp3")
```

#### Pattern 2: Direct Model Creation
```python
# Create model directly
model = FasterWhisperModel(model="base")

# Use the model
result = model.transcribe(path="audio.mp3")
```

### Multiple Transcriptions
For multiple transcriptions, load the model once and reuse it:
```python
# Load model once
model = load_model(engine="faster-whisper", model="base")

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