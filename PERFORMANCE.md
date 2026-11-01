# faster-whisper CPU impact

This document answers [#15](https://github.com/ivrit-ai/ivrit-py/issues/15):
*why does `faster-whisper` look CPU-bound even with the model on a GPU, and
what can we do about it?*  It records what was measured, what was ruled out, and
the optimization that came out of it.

Everything below is reproducible with
`benchmarks/faster_whisper_benchmark.py` (see [Reproducing](#reproducing)).

## TL;DR

- Audio decoding is **not** the bottleneck.  On a 10 s / 2 min / 10 min clip it
  is 3.5% / 2.0% / 1.8% of wall time.  Log-Mel feature extraction is under 1%.
- The CPU time comes from **CTranslate2's decoding loop**: sampling, token
  bookkeeping and the CPU/GPU synchronisation around every decode step run on
  the CPU while the GPU waits.  It is ~68% of wall time and ~80% of all CPU
  seconds.
- The one piece of *pure waste* in that loop is a **duplicate encoder pass**:
  when `language` is not pinned, faster-whisper encodes the first window to
  detect the language, then encodes the exact same window again to start
  decoding.  That is one full encoder pass per `transcribe()` call.
- `ivrit` now reuses the probe's encoder output (and the log-Mel features it was
  computed from).  Transcripts are **identical on the tested corpus**, and it is
  worth (medians of 5 interleaved runs, `language=None`):

  | audio length | speed-up |
  |---|---|
  | 1 s | 48.2% |
  | 3 s | 45.1% |
  | 7 s | 41.4% |
  | 15 s | 33.7% |
  | 30 s | 20.0% |
  | 60 s | 12.2% |
  | 90 s | 8.5% |
  | 125 s | 5.6% |
  | 10 min | ~1-2% (one ~0.24 s encoder pass; noise dominates at this length) |

  The saving is exactly one encoder pass per `transcribe()` call, so it is a
  fixed amount of time rather than a fixed percentage: >10% up to roughly a
  minute of audio, and ~1-2% on a 10 min file (one of its 23 encoder passes).  The
  encoder-pass count is deterministic: 2→1 (10 s), 6→5 (125 s), 23→22 (10 min).
  If the caller pins `language`, faster-whisper never runs the extra pass, so
  there is nothing to save (and ivrit's A/B is unchanged).

  Incremental (`WhisperSession`) transcription, which is ivrit's streaming path
  and transcribes short buffers over and over, improves by **31.8%** on a 2 min
  stream fed in 10 s chunks.
- **Does this meet the 10% target?**  For the default, exact path: yes for
  clips up to about a minute (12-48%) and for incremental sessions (31.8%).
  Long-form exact gains are bounded by one encoder pass per call, so they are
  ~1-2% on a 10 min file; no exact change can do better without changing the
  decoder loop inside CTranslate2.  For long-form throughput, faster-whisper's
  batched pipeline measured **1.26x / 1.30x** (2 min / 10 min), but it changes
  segmentation and is tracked separately in #42, not included here.

## What "CPU-bound" actually means here

Measured on a 10 min clip, `cpu / wall = 1.95`: the process keeps roughly two
cores busy for the whole transcription even though the model is on CUDA.

The per-stage breakdown shows where those seconds are (10 min clip, RTX 4060
Laptop 8 GB, `float16`, `word_timestamps=True`, language auto-detected):

| stage | wall | cpu | % of wall |
|---|---|---|---|
| decoder + sampling (GPU/CPU) | 12.94 s | 29.41 s | 68.3% |
| encoder (GPU) | 5.22 s | 5.91 s | 27.5% |
| audio decode (PyAV) | 0.35 s | 0.35 s | 1.8% |
| word alignment | 0.27 s | 0.56 s | 1.4% |
| language detection (nested in encoder) | 0.23 s | 0.93 s | 1.2% |
| log-Mel features | 0.16 s | 0.78 s | 0.9% |

The same shape holds for shorter inputs: on a 10 s clip the two encoder passes
are 66.6% of wall time, decoding 27.7%, and audio decoding 3.5%.

So the CPU load is **inside the decoder loop**, not in the audio path:

- every decode step is a GPU forward pass plus CPU-side sampling/bookkeeping and
  a synchronisation, so the GPU idles between steps.  GPU utilisation during a
  10 min transcription averages 67% (76% while active), and running two
  transcriptions concurrently does not increase throughput (29x realtime either
  way) — the GPU is the limiter, with gaps that the CPU work creates.
- the decoder dominates CPU seconds (29.4 s of the 37.0 s total).

## What was ruled out

| candidate | measurement | verdict |
|---|---|---|
| audio decoding | 1.8–3.5% of wall, 1 core | not the bottleneck |
| log-Mel features | 0.3–0.9% of wall | not the bottleneck |
| word timestamps | 1.0–1.4% of wall | not the bottleneck |
| `cpu_threads` too high | default (4) is the wall-time optimum; 8 keeps the same wall time but burns 60% more CPU; 1 is 21% slower | keep the default |
| parallel requests | 2 concurrent workers: same aggregate throughput as 1 | no win on a power-limited GPU |
| `compute_type=int8_float16` | encoder 10% faster | accuracy risk; the engine already avoids int8 deliberately |
| beam / prompt tweaks | `beam_size=1` is 29% faster but changes text (similarity 0.963); `condition_on_previous_text=False` is 6% faster and also changes text | not acceptable as defaults |

One more experiment is worth calling out: restricting the process to 2, 4, 8 or
16 cores leaves wall time unchanged on a 2 min file (4.11–4.14 s).  That is the
clearest statement of the situation — for long files the pipeline is GPU-bound,
so shaving CPU work in the audio path cannot produce a meaningful speed-up.  Any
real win has to remove *GPU* work.  The duplicate encoder pass is exactly that.

## The duplicate encoder pass

`WhisperModel.transcribe()` does this when `language` is `None`:

1. `detect_language()` runs `self.encode(pad_or_trim(features[..., :3000]))` to
   classify the language of the first window.
2. `generate_segments()` starts at `seek == 0` with `encoder_output is None`, so
   it runs `self.encode(segment)` on the *same* window before decoding.

Counting encoder calls confirms it: a 2 min file takes 6 encoder passes for 5
windows with `language=None`, and 5 passes when `language="he"` is passed.  On a
10 s clip it is 2 passes for 1 window, and on a 10 min file 23 passes for 21
windows; the optimized path always does exactly one fewer.

`ivrit/_fw_perf.py` removes the duplicate:

- decode the source once (as faster-whisper would) and compute its log-Mel
  features once;
- build the first window exactly the way `generate_segments()` does;
- run it through the encoder once and read the language from that output with
  the same token parsing faster-whisper uses;
- call `transcribe()` with that language, and hand the cached encoder output
  back when `generate_segments()` asks for the same window (the cached features
  are reused too).

faster-whisper's `generate_segments()` already accepts an `encoder_output`, but
`transcribe()` does not expose it; the adapter is the minimal way to use it
without forking faster-whisper.  If upstream exposes the probe output, the
adapter can be deleted.

For clips of 30 s or longer the encoder sees exactly the window the decoder
would, so this cannot change the transcript.  The A/B runs in the table above
compare every segment, word and timestamp and report `identical=True` at every
length from 1 s to 125 s, and the `info` returned by `transcribe()` keeps the
detected `language_probability` / `all_language_probs` instead of the `1.0` /
`None` that pinning a language would produce.

One detail for clips shorter than 30 s: faster-whisper's probe window includes
one Mel frame (10 ms) that its decoding window drops, so the two windows are not
byte-identical there.  The adapter probes with the decoding window instead, which
is the only way to reuse the pass; on the test corpus this changes neither the
detected language nor a single token, though `language_probability` can differ
slightly at that length.  From 30 s up the two windows are identical by
construction.

The adapter works on a shallow copy of the `WhisperModel` whose `encode` and
`feature_extractor` attributes are shadowed, so the caller's model is never
mutated and concurrent callers cannot observe each other's override.  It falls
back to the plain call whenever the first window would differ (VAD,
`clip_timestamps`, `chunk_length`, `multilingual`, `language_detection_segments >
1`) or when faster-whisper's internals do not look the way the adapter expects.

### Incremental / streaming transcription

`WhisperSession` re-transcribes a short buffer on every `append()`, so it pays
the language probe on every chunk.  Feeding `tests/asimov.mp3` (125 s) in 10 s
PCM chunks:

| | wall | cpu | segments |
|---|---|---|---|
| probe reuse disabled | 10.04 s | 25.08 s | 32 |
| probe reuse enabled | 6.85 s | 21.85 s | 32 |

**31.8% faster**, identical segments.  Both rows include the session fix below;
the comparison isolates the encoder-pass reuse.

While measuring this path, `WhisperSession` turned out to be raising on every
`append()` on `master`: `transcribe_core()` requires `output_options`, which the
session never passed, so the exception was swallowed and sessions silently
returned no segments.  That is fixed here as well, with a regression test.

## The remaining long-form lever

The remaining lever for long files is to decode several windows per forward
pass, which amortises the per-step CPU work and keeps the GPU busy.
faster-whisper ships this as `BatchedInferencePipeline` (it uses the bundled
Silero VAD to group windows).  It is **not enabled by this change**: windows are
no longer conditioned on the previous window's text and segment boundaries follow
VAD chunks, so transcripts are close but not identical.  It is tracked in #42.

| audio | sequential | `batch_size=8` | speed-up | word counts | text similarity |
|---|---|---|---|---|---|
| 2 min | 4.10 s | 3.24 s | 1.26x | 219 / 218 | 0.962 |
| 10 min | 18.96 s | 14.55 s | 1.30x | 1097 / 1094 | 0.932 |

## Reproducing

```bash
# stage-by-stage wall/CPU breakdown
python benchmarks/faster_whisper_benchmark.py stages tests/asimov.mp3

# default path vs vanilla faster-whisper (checks output equality)
python benchmarks/faster_whisper_benchmark.py ab tests/asimov.mp3

# speed-up as a function of audio length
python benchmarks/faster_whisper_benchmark.py lengths tests/asimov.mp3

# incremental session path
python benchmarks/faster_whisper_benchmark.py session tests/asimov.mp3

# faster-whisper's batched pipeline, for comparison (#42)
python benchmarks/faster_whisper_benchmark.py batch tests/asimov.mp3
```

The `lengths` and `session` modes shell out to `ffmpeg`/`ffprobe`, so both must
be on `PATH`.

The numbers in this document were produced with:

- Ryzen 7 7840HS (8C/16T), 16 GB RAM
- NVIDIA RTX 4060 Laptop GPU, 8 GB, 60 W power limit
- CUDA 12.9 / cuDNN 9.27 (from the `nvidia-*` wheels)
- Python 3.12, `faster-whisper==1.2.1`, `ctranslate2==4.8.4`
- `ivrit-ai/whisper-large-v3-turbo-ct2`, `device=cuda`, `compute_type=float16`,
  `word_timestamps=True`, `language=None`

If CUDA libraries are installed outside the default loader path, point
`LD_LIBRARY_PATH` at them before running (e.g. the `nvidia/*/lib` directories of
the virtualenv) so CTranslate2 can load `libcublas`/`libcudnn`.

## Limitations

- Gains are a fixed amount of work (one encoder pass) rather than a percentage,
  so they shrink as audio gets longer: ~0.24 s per `transcribe()` call on this
  GPU.  That is 26-48% on clips under 30 s, >10% up to about a minute, and ~1-2%
  on a 10 min file.
- The optimization only applies when `language` is left `None`; pinning it
  already avoids the extra pass.  It also does not apply to very short audio
  that produces no full encoder frame (falls back to the plain call).
- Measured on one GPU class.  The saving scales with encoder cost, so faster
  GPUs will see a similar percentage (the encode pass shrinks together with the
  rest of the GPU work), while CPU-only runs see less.
- The batched pipeline measured here is a throughput trade-off, not an
  equivalence-preserving optimization, and is not enabled by this change.
