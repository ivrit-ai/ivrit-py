# ivrit-py Architecture

`ivrit` is a Python package that wraps multiple speech-to-text engines and speaker
diarization backends behind a single, uniform API. The public surface lives in
`ivrit/__init__.py` and exposes `load_model`, `TranscriptionModel`,
`TranscriptionSession`, and `Segment`.

## Package Layout

```
ivrit/
├── __init__.py        # Public API re-exports
├── audio.py           # Transcription models, sessions, and load_model factory
├── diarization.py     # Speaker diarization engines and the diarize() entry point
├── types.py           # Shared dataclasses: Word, Segment
└── utils.py           # Audio I/O, dependency checks, device detection
```

Optional heavy dependencies (`torch`, `faster-whisper`, `pyannote-audio`,
`speechbrain`, `pywhispercpp`, `stable-ts-whisperless`, `numpy`, `pandas`,
`scikit-learn`, ...) are declared under the `[all]` extra and imported lazily
through `utils.check_dependencies` so that the core package stays installable
without them.

## Core Types (`ivrit/types.py`)

- **`Word`** — single token with `word`, `start`, `end`, optional `probability`
  and `speaker`.
- **`Segment`** — transcription segment with `text`, `start`, `end`, a list of
  `speakers`, a list of `Word`s, and a free-form `extra_data` dict. `__post_init__`
  rehydrates dict-form words into `Word` instances so segments round-trip cleanly
  through JSON.

These are the lingua franca exchanged between transcription and diarization
layers. Engine-specific data that doesn't fit the schema lives in
`Segment.extra_data`.

## Transcription Layer (`ivrit/audio.py`)

### Abstractions

- **`TranscriptionModel`** (ABC) — the engine-agnostic interface. Concrete
  subclasses implement engine-specific transcription. Defines:
  - `transcribe(*, path|url|blob, language, stream, diarize, diarization_args,
    output_options, verbose, **kwargs)` — sync entry point returning either a
    `dict` or a `Generator[Segment]` depending on `stream`.
  - `transcribe_async(...)` — async variant returning an `AsyncGenerator` /
    awaitable.
  - `create_session(...)` — optional, raises `NotImplementedError` by default.
    Only engines that natively support incremental decoding override it.
- **`TranscriptionSession`** (ABC) — incremental, stateful transcription. Methods:
  `append(audio_bytes)`, `get_all_segments()`, `get_full_text()`,
  `get_session_info()`, `reset()`, `flush()`. Sessions consume raw mono s16le PCM
  at the session's `sample_rate`.

### Concrete Engines

| Class                | Engine name      | Backend                              | Sessions |
|----------------------|------------------|--------------------------------------|----------|
| `FasterWhisperModel` | `faster-whisper` | `faster_whisper.WhisperModel`        | yes (via `WhisperSession`) |
| `StableWhisperModel` | `stable-whisper` | `stable_whisper` (whisperless build) | yes (via `WhisperSession`) |
| `WhisperCppModel`    | `whisper-cpp`    | `pywhispercpp.model.Model`           | no       |
| `RunPodModel`        | `runpod`         | RunPod-hosted endpoint over HTTP     | no       |

- **`WhisperSession`** is the shared session implementation reused by both
  faster-whisper and stable-whisper. It buffers PCM frames and emits segments
  with confidence-based filtering, deferring final segments until `flush()`.
- **`RunPodJob` / `AsyncRunPodJob`** are the sync/async polling helpers that wrap
  a RunPod inference job and stream segments back as they become available.
- **`_copy_segment_extra_data`** is the shared helper that pulls all
  JSON-serializable, non-core attributes off backend-native segments into
  `Segment.extra_data`, so engine-specific metadata is preserved without
  polluting the core schema.
- **`get_device_and_index`** parses device strings like `"cuda:1"` into a
  `(device, index)` pair for the underlying libraries.

### Factory

`load_model(*, engine, model, **kwargs) -> TranscriptionModel` is the single
public entry point for instantiation. It dispatches on `engine` to the matching
class and forwards `kwargs` through to the underlying constructor. Unknown
engines raise `ValueError`.

### Diarization Hook

When `transcribe(..., diarize=True)` is called, the model performs
transcription, then delegates to `ivrit.diarization.diarize()` with the
collected segments and the original audio. `diarization_args` is passed through
verbatim and selects the diarization engine and its parameters.

## Diarization Layer (`ivrit/diarization.py`)

### Abstractions

- **`BaseDiarizationEngine`** (ABC) — defines `diarize(audio,
  transcription_segments, *, device, ...) -> List[Segment]`. Implementations
  mutate the provided segments in place to attach speaker labels and also
  return them.

### Concrete Engines

- **`PyannoteDiarizationEngine`** — wraps the `pyannote.audio` neural pipeline.
  Loads a checkpoint (default or user-supplied), runs it over the audio, then
  uses `_match_speaker_to_interval` / `_assign_speakers` to attach pyannote's
  speaker turns to each `Segment`.
- **`IvritDiarizationEngine`** — embedding-based pipeline using SpeechBrain
  ECAPA-TDNN. The flow is:
  1. `_load_audio_speechbrain` decodes the audio via ffmpeg (the project
     deliberately avoids torchaudio because it can't handle every container).
  2. `_extract_segment_audio` slices per-segment waveforms.
  3. ECAPA-TDNN produces speaker embeddings for each slice.
  4. `_try_clustering_methods` runs several clustering algorithms across a
     range of cluster counts.
  5. `_calculate_clustering_metrics` and `_calculate_composite_score` rank the
     candidates so the best partition wins automatically.
  6. `_process_clustering_results` and `_assign_speakers_to_all_segments`
     attach the resulting speaker labels back to the segments.

### Public Entry Point

`diarize(audio, transcription_segments, *, engine, ...) -> List[Segment]` is
the canonical, engine-dispatching function. It validates `engine` against
`{"pyannote", "ivrit"}` and forwards the engine-relevant subset of arguments to
the matching engine instance.

## Utilities (`ivrit/utils.py`)

- **`check_dependencies(module_specs, feature_name)`** — lazy import helper.
  All optional dependencies are pulled in through this so that import-time
  failures become actionable error messages pointing at `pip install ivrit[all]`.
- **`get_audio_file_path(path|url|blob)`** — normalizes the three accepted
  audio sources into a local filesystem path. URL and blob inputs are
  materialized into temp files; the caller owns cleanup.
- **`load_audio(file, sr)`** — shells out to `ffmpeg` to decode arbitrary
  containers into mono float32 PCM at the requested sample rate. ffmpeg is the
  intentional decode path everywhere; torchaudio is avoided because it does not
  handle every input format.
- **`guess_device()`** — returns `"cuda"`, `"mps"`, or `"cpu"` based on what
  torch reports as available.
- **`SAMPLE_RATE = 16000`** — package-wide canonical sample rate.

## End-to-End Flow

1. Caller imports `ivrit` and calls `load_model(engine=..., model=..., ...)`.
2. The factory returns a concrete `TranscriptionModel`.
3. Caller invokes `model.transcribe(path=..., diarize=True, ...)`.
4. The engine resolves the audio source via `utils.get_audio_file_path`,
   transcribes it (possibly streaming), normalizes results into `Segment`
   objects, and (if requested) calls `diarization.diarize()` to attach speaker
   labels.
5. For incremental use cases, the caller instead does
   `session = model.create_session(...)`, feeds raw PCM via `session.append`,
   and finalizes with `session.flush()`.

## Tests and Examples

- `tests/` contains pytest suites: `test_basic_transcription.py`,
  `test_diarization.py`, `test_session.py`, `test_transcribe_async.py`, plus
  fixture audio (`asimov.mp3`, `test_input_10s.mp3`).
- `examples/` is currently empty and reserved for runnable usage samples.
