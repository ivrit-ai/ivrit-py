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
    output_options, verbose, on_progress, **kwargs)` — sync entry point. Each of
    `path`/`url`/`blob` accepts `Union[str, List[str]]`. A bare string selects
    single-file mode and returns either a `dict` or a `Generator[Segment]`
    depending on `stream` (byte-for-byte unchanged behavior). A list — even a
    one-element list like `path=["a.mp3"]` — selects **batch mode** and drives
    the return type: `stream=False` returns `List[dict]` (one entry per input,
    in input order; a failed item is isolated as
    `{"error": str, "source": kind, "input": value}`), and `stream=True` returns
    `Generator[Tuple[int, Union[Segment, Exception]]]` yielding `(index, Segment)`
    and, on per-item failure, a single `(index, exc)` before continuing. Sources
    are mutually exclusive and a batch is homogeneous (no mixing kinds); an empty
    list raises `ValueError`. Batch items are processed strictly sequentially.
  - `transcribe_async(...)` — async variant; **always streams**. Single-file mode
    yields `Segment` objects; batch mode (list source) yields
    `AsyncGenerator[Tuple[int, Union[Segment, Exception]]]` with the same
    `(index, Segment)` / `(index, exc)` shape and per-item isolation. Internally
    delegates to `_transcribe_one_async(...)`, the single-source async seam
    (RunPod overrides `_transcribe_one_async` with its native aiohttp
    implementation; the batch-aware `transcribe_async` is the non-overridden base
    that fans single-source calls out sequentially). The sync batch path likewise
    funnels each item through the single-source `_transcribe_one(...)` helper.
  - `create_session(...)` — optional, raises `NotImplementedError` by default.
    Only engines that natively support incremental decoding override it.
  - `_transcribe_batch(...)` / `_transcribe_batch_async(...)` — the overridable
    **batch seam**. After `transcribe` / `transcribe_async` eagerly validate the
    sources, the batch branch delegates to these methods (single-file calls stay
    on the direct `_transcribe_one` / `_transcribe_one_async` path). The default
    bodies are the per-item fan-out loop (sequential, per-item error isolation,
    per-item `_wrap_progress` attribution). `RunPodModel` overrides both to
    submit a remote job carrying a list payload and demux the index-tagged
    stream instead of issuing N single-source jobs. A URL batch, and a
    blob/path batch that fits the payload cap, run as **one** job; a blob/path
    batch whose combined payload exceeds the cap is split by a greedy in-order
    chunk planner into **N sequential jobs**, each remapping its worker-local
    indices back to caller global indices so the caller contract is identical
    regardless of job count. The seams own the per-item progress wrapping;
    source normalization stays in the public methods.

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
  diarization phase are listed in the Diarization Layer section below. In batch
  mode the `on_progress` callback is wrapped per item so each event's `extra`
  dict additionally carries `batch_index` and `batch_total` (merged without
  clobbering engine-supplied extras); the four core fields are unchanged. Single
  (non-batch) calls pass the callback through untouched.
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
- **`FasterWhisperModel`** pre-decodes audio files through `utils.load_audio()`
  and passes the resulting mono 16 kHz float32 waveform directly to
  faster-whisper. This avoids faster-whisper's file decoding path in the common
  local-file flow. If local ffmpeg-based decoding is unavailable, it logs a
  warning and falls back to passing the file path to faster-whisper.
- **`RunPodJob` / `AsyncRunPodJob`** are the sync/async polling helpers that wrap
  a RunPod inference job and stream results back as they become available.
  The RunPod stream protocol carries `output` entries inside `data['stream']`,
  each tagged with a `type` (`segments`, `progress`, or `error`) and an optional
  per-entry `index` used for batch demux. When `index` is absent (single-source
  mode) the helper yields bare values for full back-compat — a bare `Segment`
  for `segments`, a bare `{"progress": ...}` for `progress`, and a bare
  `Exception` for `error` (so single-mode callers' existing raise path trips).
  When `index` is present (batch mode) the same three kinds are yielded as
  `(index, Segment)`, `{"progress": ..., "index": index}`, and
  `(index, Exception)` respectively. The orchestrator (`_run_job_stream` /
  `_run_job_stream_async`) normalizes both shapes into `(index, item)` and routes
  progress to `on_progress`; per-index worker errors surface as
  `(index, Exception)` without aborting the rest of the stream.
  RunPod's `/stream` endpoint reports `COMPLETED` even for jobs that have failed,
  and for jobs that have not started yet, so its status field is never taken as
  the job's outcome. Whenever a `/stream` response carries no further data the
  helpers confirm the real outcome through `status_body()` (`/status`, the only
  place RunPod reports a failure's `error`): a still-running job means streaming
  continues, a failed one raises carrying that error. Without this a failed job —
  a bad model name, say — ends the stream silently and is indistinguishable from
  a successful transcription of a silent recording.
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
5. When a source argument is a list, the call enters the batch seam
   (`_transcribe_batch` / `_transcribe_batch_async`). For local engines the
   default seam fans out sequentially: each item is funneled through the same
   single-file path (sync via `_transcribe_one`, async via
   `_transcribe_one_async`) in input order, with results attributed by their
   input index and per-item errors isolated rather than aborting the batch.
   `RunPodModel` overrides the seam to carry the list as a single payload whose
   `transcribe_args` source key is a list, and demuxes the worker's
   index-tagged stream back into per-item results. `RUNPOD_MAX_PAYLOAD_LEN` is
   measured as JSON UTF-8 bytes (not `str(payload)`, which undercounts
   non-ASCII like Hebrew) and is enforced before submission. URL batches never
   split: they stay small and scale well, so the whole list is one job. A
   blob/path batch over the cap is split by a greedy in-order chunk planner
   (`_plan_blob_chunks`) into multiple sequential jobs that each fit; each
   chunk's worker-local indices (0..k-1) are remapped back to caller global
   indices, so the caller-visible streaming and non-streaming contracts are
   identical whether 1 or N jobs ran. A single blob/path element that alone
   exceeds the cap still raises `ValueError`. Error
   handling is deliberately asymmetric: a per-item worker error is isolated as
   `(index, Exception)` (streaming) or `{"error": ...}` (non-streaming), whereas
   a whole-job failure (network/queue `TimeoutError`, payload-too-large
   `ValueError`) propagates from the batch call.
6. For incremental use cases, the caller instead does
   `session = model.create_session(...)`, feeds raw PCM via `session.append`,
   and finalizes with `session.flush()`.

## Tests and Examples

- `tests/` contains pytest suites: `test_basic_transcription.py`,
  `test_diarization.py`, `test_session.py`, `test_transcribe_async.py`, plus
  fixture audio (`asimov.mp3`, `test_input_10s.mp3`).
- `examples/` is currently empty and reserved for runnable usage samples.
