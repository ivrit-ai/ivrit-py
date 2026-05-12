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
    output_options, verbose, on_progress, **kwargs)` — sync entry point
    returning either a `dict` or a `Generator[Segment]` depending on `stream`.
  - `transcribe_async(...)` — async variant returning an `AsyncGenerator` /
    awaitable.
  - `create_session(...)` — optional, raises `NotImplementedError` by default.
    Only engines that natively support incremental decoding override it.

  All transcription entry points accept an optional `on_progress:
  Callable[[dict], None]` callback. It is invoked periodically as work
  advances and receives a dict with four core fields that are always
  present: `phase` (`"transcription"` or `"diarization"`), `step` (str,
  a sub-phase label such as `"decode"`, `"embedding"`, or `"clustering"`),
  `step_fraction` (float, 0.0--1.0, representing progress within the
  current step; 0.0 when the engine cannot compute a fraction), and
  `description` (str, a short human-readable label suitable for a UI
  progress indicator, e.g. `"Transcribing audio"` or
  `"Diarization: clustering speakers"`). An `extra` dict contains
  engine-specific data (e.g. `processed_seconds`, `total_seconds`,
  `segment_index`, `clusters_tried`). `processed_seconds` and
  `total_seconds` are no longer core fields — they live inside `extra`
  and are only present when the engine can provide them. Exceptions
  raised by the callback are caught and logged at warning level so a
  faulty callback never aborts a run. The callback is wired uniformly
  across every engine — and through both the transcription and
  diarization phases — via the `emit_progress` / `invoke_progress`
  helpers in `ivrit/utils.py`. Each engine emits progress from whichever
  native hook its backend exposes (faster-whisper: per-yielded-segment
  with `info.duration`; stable-whisper: native `progress_callback`;
  whisper-cpp: native `new_segment_callback`; runpod: `progress` items
  on the worker stream protocol). Per-engine details for the
  diarization phase are listed in the Diarization Layer section below.
- **`TranscriptionSession`** (ABC) — incremental, stateful transcription. Methods:
  `append(audio_bytes)`, `get_all_segments()`, `get_full_text()`,
  `get_session_info()`, `reset()`, `flush()`. Sessions consume raw mono s16le PCM
  at the session's `sample_rate`.

### Concrete Engines

| Class                | Engine name      | Backend                              | Session support |
|----------------------|------------------|--------------------------------------|-----------------|
| `FasterWhisperModel` | `faster-whisper` | `faster_whisper.WhisperModel`        | Yes |
| `StableWhisperModel` | `stable-whisper` | `stable_whisper` (whisperless build) | Yes |
| `WhisperCppModel`    | `whisper-cpp`    | `pywhispercpp.model.Model`           | No  |
| `RunPodModel`        | `runpod`         | RunPod-hosted endpoint over HTTP     | No  |

- **`WhisperSession`** is the shared session implementation reused by both
  faster-whisper and stable-whisper. It buffers PCM frames and emits segments
  with confidence-based filtering, deferring final segments until `flush()`.
- **`FasterWhisperModel` batched path** — when `transcribe(..., batch_size=N)`
  is called with `N > 1`, the wrapper switches from
  `faster_whisper.WhisperModel.transcribe` to
  `faster_whisper.BatchedInferencePipeline.transcribe` while keeping the
  public `ivrit` API unchanged. Additional transcription kwargs such as
  `beam_size`, `vad_filter`, and `batch_size` are forwarded to the
  faster-whisper backend call.
- **`RunPodJob` / `AsyncRunPodJob`** are the sync/async polling helpers that wrap
  a RunPod inference job and stream results back as they become available.
  The RunPod stream protocol carries two kinds of items inside `data['stream']`:
  `output` items (lists of segment dicts that are reconstructed into `Segment`
  objects and yielded) and `progress` items (free-form dicts that are yielded
  as `{"progress": ...}` and routed by the orchestrator to `on_progress`).
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
verbatim and selects the diarization engine and its parameters. The same
`on_progress` callback that was supplied to `transcribe()` is forwarded to
`diarize()` so the consumer receives a continuous stream of events with
`phase` transitioning from `"transcription"` to `"diarization"`.

## Diarization Layer (`ivrit/diarization.py`)

### Abstractions

- **`BaseDiarizationEngine`** (ABC) — defines `diarize(audio,
  transcription_segments, *, device, on_progress, ...) -> List[Segment]`.
  Implementations mutate the provided segments in place to attach speaker
  labels and also return them. Each implementation may call `emit_progress`
  with `phase="diarization"` and engine-specific data in `extra` at natural
  progress points.

### Concrete Engines

- **`PyannoteDiarizationEngine`** — wraps the `pyannote.audio` neural pipeline.
  Loads a checkpoint (default or user-supplied), runs it over the audio, then
  uses `_match_speaker_to_interval` / `_assign_speakers` to attach pyannote's
  speaker turns to each `Segment`. Progress is emitted by passing a small
  hook adapter as `pipeline(..., hook=...)` that translates pyannote's
  `(step_name, step_artifact, file, total, completed)` protocol into
  `phase="diarization"` events with `step` set to the pyannote step name
  (e.g. `"segmentation"`, `"embeddings"`), `step_fraction` derived from
  `completed/total`, and `extra` containing `step_completed` and
  `step_total`.
- **`IvritDiarizationEngine`** — embedding-based pipeline using SpeechBrain
  ECAPA-TDNN. The flow is:
  1. `_load_audio_speechbrain` decodes the audio via ffmpeg (the project
     deliberately avoids torchaudio because it can't handle every container).
  2. `_extract_segment_audio` slices per-segment waveforms.
  3. ECAPA-TDNN produces speaker embeddings for each slice. **One progress
     event per segment** with `step="embedding"`,
     `step_fraction=(i+1)/len(segments)`, and `extra` containing
     `segment_index`, `segment_total`, `skipped`,
     `processed_seconds=segment.end`, and
     `total_seconds=max(segment.end for segments)`.
  4. `_try_clustering_methods` runs several clustering algorithms across a
     range of cluster counts. **One progress event per `n_clusters` value
     tried** with `step="clustering"` and `extra` containing
     `clusters_tried`, `clusters_total`, `n_clusters`.
  5. `_calculate_clustering_metrics` and `_calculate_composite_score` rank the
     candidates so the best partition wins automatically.
  6. `_process_clustering_results` and `_assign_speakers_to_all_segments`
     attach the resulting speaker labels back to the segments.

### Public Entry Point

`diarize(audio, transcription_segments, *, engine, ..., on_progress) ->
List[Segment]` is the canonical, engine-dispatching function. It validates
`engine` against `{"pyannote", "ivrit"}` and forwards the engine-relevant
subset of arguments — including `on_progress` — to the matching engine
instance. The `on_progress` contract is the same one used by
`TranscriptionModel.transcribe`, so a single callback can serve both phases
of a transcribe-then-diarize run.

## Utilities (`ivrit/utils.py`)

- **`ProgressCallback`** type alias and the **`emit_progress` /
  `invoke_progress`** helpers — the shared plumbing behind the unified
  `on_progress` contract used across every transcription engine and every
  diarization engine. `invoke_progress` calls a user-supplied callback with
  a pre-built dict and swallows exceptions (logged at `warning`);
  `emit_progress` is a thin builder around it that accepts the four core
  fields (`phase`, `step`, `step_fraction`, `description`) as keyword
  arguments and nests any additional `**extras` under an `"extra"` key in
  the emitted dict.
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
