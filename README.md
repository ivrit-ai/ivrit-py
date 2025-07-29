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

### Using uv

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package and project management. uv provides a unified interface for managing Python versions, dependencies, and virtual environments.

#### Installing uv

Install uv using the official installer:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

#### Environment Setup

uv automatically manages virtual environments and dependencies. To set up the development environment:

```bash
# Clone the repository
git clone <repository-url>
cd ivritu

# uv will automatically create a virtual environment and install dependencies
uv sync --all-extras
```

All development commands should be run with `uv run` prefix to ensure they use the correct environment and dependencies.

### Installation for Development

With uv (recommended):

```bash
git clone <repository-url>
cd ivrit
uv sync --all-extras
```

Or with traditional pip:

```bash
git clone <repository-url>
cd ivrit
pip install -e ".[dev]"
```

### Pre-commit Hooks

This project uses pre-commit hooks for code quality and formatting. Install them with:

```bash
uv run pre-commit install
```

The hooks will automatically run on each commit and include:
- End-of-file fixing
- Large file checks
- YAML and JSON validation
- Code linting and formatting with Ruff
- TOML formatting with Taplo

To run the hooks manually on all files:

```bash
uv run pre-commit run --all-files
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

Code formatting and linting are handled automatically by pre-commit hooks using Ruff. To run manually:

```bash
# Check and fix linting issues
uv run ruff check --fix

# Format code
uv run ruff format
```

## License

MIT License - see LICENSE file for details. 
