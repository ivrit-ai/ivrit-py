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
```

#### Async Transcription (RunPod Only)

For RunPod models, you can use async transcription for better performance:

```python
import asyncio
from ivrit.audio import load_model

async def transcribe_async():
    # Load RunPod model
    model = load_model(
        engine="runpod",
        model="large-v3-turbo",
        api_key="your-api-key",
        endpoint_id="your-endpoint-id"
    )
    
    # Stream results asynchronously
    async for segment in model.transcribe_async(path="audio.mp3", language="he"):
        print(f"{segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")

# Run the async function
asyncio.run(transcribe_async())
```

**Note**: Async transcription is only available for RunPod models. The sync `transcribe()` method uses the original sync implementation.

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

### `transcribe()` and `transcribe_async()`

Transcribe audio using the loaded model.

#### Parameters

- **path** (`str`, optional): Path to the audio file to transcribe
- **url** (`str`, optional): URL to download and transcribe
- **blob** (`str`, optional): Base64 encoded blob data to transcribe
- **language** (`str`, optional): Language code for transcription (e.g., 'he' for Hebrew, 'en' for English)
- **stream** (`bool`, optional): Whether to return results as a generator (True) or full result (False) - only for `transcribe()`
- **diarize** (`bool`, optional): Whether to enable speaker diarization
- **verbose** (`bool`, optional): Whether to enable verbose output
- **\*\*kwargs**: Additional keyword arguments for the transcription model

#### Returns

- `transcribe()`: If `stream=True`: Generator yielding transcription segments, If `stream=False`: Complete transcription result as dictionary
- `transcribe_async()`: AsyncGenerator yielding transcription segments

#### Raises

- `ValueError`: If multiple input sources are provided, or none is provided
- `FileNotFoundError`: If the specified path doesn't exist
- `Exception`: For other transcription errors

**Note**: `transcribe_async()` is only available for RunPod models and always returns an AsyncGenerator.

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

# Bounty rules

Like our bounties, and want to help?
Here's how this works:

1. You pick a bounty you're interested in, and let us know. We discuss it together to make sure you understand the issue.
2. You let us know you're on it; we lock it for you for 2 weeks so you can develop, review and merge your code.
3. The ONLY metric for whether you met the bounty goal is whether we decide to merge your PR.
   Our key focus with reviews is to ensure high code and product quality.
4. Once your PR is merged, you receive the bounty award.

You can use any tool you'd like to write your code, including AI.
Note that during review you will be asked questions about the code; if you are unable to explain what it does, or how (sometimes the case when doing Vibe coding), your PR will be discarded and you will not be able to reapply for this issue.

## Design Notes

- [Word-level streaming options](docs/word_level_streaming_options.md)

Reviews may be done live.

## License

MIT License - see LICENSE file for details. 
