# Word-Level Streaming Options

This document analyzes practical options for word-level streaming in `ivrit`.
The focus is Whisper-family backends and, in particular, repeated transcription
of a rolling audio buffer with confidence and stability analysis.

## Current State

`ivrit` currently exposes two streaming-related surfaces:

- `transcribe(..., stream=True)` yields completed `Segment` objects from a
  full transcription run.
- `create_session(...).append(...)` buffers raw PCM and repeatedly transcribes
  the accumulated audio. The shared `WhisperSession` emits all but the final
  segment and keeps the trailing uncertain audio in the buffer.

That session design is already close to the most portable approach for
word-level streaming because it does not depend on a backend-specific decoder
state API. The missing piece is a stability policy that compares repeated
transcription results and emits words only after they are unlikely to change.

## Backend Options

| Option | Word timestamps | Partial-result support | Fit for ivrit |
| --- | --- | --- | --- |
| `faster-whisper` | Yes, with `word_timestamps=True` | No stable incremental decoder API in the current wrapper | Best default for repeated-buffer streaming |
| `stable-whisper` / `stable-ts` | Strong word timing and alignment tools | Mostly full-window transcription with better timestamp stabilization | Best quality path when latency budget allows |
| `whisper.cpp` via `pywhispercpp` | Segment callbacks are available; basic API does not expose word timestamps reliably | Callback-oriented, but less aligned with ivrit's `Word` model today | Useful for low-resource devices, weaker for word-level API |
| Remote RunPod worker | Depends on worker implementation | Existing stream protocol can relay worker progress and output | Good deployment path after local policy is settled |

## Repeated Transcription Strategy

The most portable implementation is to run the selected Whisper backend over a
rolling PCM buffer on each `append()` call, then compare the new word sequence
against prior hypotheses.

This is the same broad approach used by Whisper-Streaming: Whisper itself is an
offline model, so streaming systems repeatedly decode growing audio chunks and
commit only the stable prefix. The relevant policy is usually called local
agreement: if consecutive decodes agree on a prefix, that prefix can be emitted.

Recommended loop:

1. Buffer incoming mono s16le PCM.
2. Skip decoding until at least 0.5-1.0 seconds of audio is available.
3. Decode the buffered audio with word timestamps enabled when the backend
   supports them.
4. Normalize the result into a flat list of candidate words:
   `text`, `start`, `end`, `probability`, and source segment index.
5. Compare the candidates with the previous decode's candidates.
6. Emit only the longest stable prefix.
7. Trim audio through the end time of the last emitted word, keeping a small
   overlap so Whisper has enough context for the next decode.

The stable-prefix rule should be text-first and time-second. Word timings can
move even when the word identity is stable, so exact timestamp equality should
not be required.

## Stability Heuristics

A word should be considered stable when all of the following are true:

- The normalized word text appears at the same prefix position in at least two
  consecutive decode passes.
- The word end time is not within the last `tail_guard_seconds` of the current
  buffer. A practical starting value is 0.5-0.8 seconds.
- If the backend provides probability, the probability is above a configurable
  threshold. A practical starting value is 0.45-0.60 for individual words.
- The word's start and end times have not shifted by more than a tolerance
  across passes. A practical starting value is 120-200 ms.

For Hebrew and other languages with attached punctuation or clitics, normalize
only for stability comparison. The emitted word should preserve the backend's
original text and timing.

## Confidence Model

Whisper confidence should be treated as advisory, not authoritative. A robust
score can combine:

- backend word probability, when present;
- number of consecutive passes with matching text;
- timestamp drift between passes;
- distance from the active buffer tail;
- whether punctuation changed while the word stayed the same.

Example score:

```text
stable_score =
  0.45 * min(1.0, consecutive_matches / 3)
  + 0.25 * probability_or_default
  + 0.20 * timestamp_stability
  + 0.10 * tail_distance_score
```

Emit when `stable_score >= 0.75`, or when flushing at the end of the stream.

## API Recommendation

Add an opt-in word streaming layer rather than changing `transcribe(stream=True)`
semantics. That keeps the existing segment stream compatible.

Suggested session API:

```python
session = model.create_session("call-123", language="he")
session.append(pcm_bytes)
for word in session.get_new_words():
    print(word.word, word.start, word.end, word.probability)
```

Suggested lower-level return type:

```python
@dataclass
class StreamingWord:
    word: str
    start: float
    end: float
    probability: float | None
    stable_score: float
    is_final: bool
```

The existing `Word` dataclass can also be reused if avoiding a new public type is
preferred. In that case, stability metadata can live in `extra_data` on emitted
segments, but a dedicated `StreamingWord` is cleaner for user interfaces.

## Existing Alternatives

### LocalAgreement / Whisper-Streaming

Whisper-Streaming is the strongest architecture reference for ivrit because it
wraps full-sequence Whisper-like models without requiring model internals. It
keeps an audio buffer, runs repeated decodes, compares the latest hypothesis with
previous hypotheses, and commits the stable prefix. This maps directly to
`WhisperSession`.

The main adaptation for ivrit is to operate at `Word` granularity rather than
only text/segment granularity.

### Forced Alignment

WhisperX-style forced alignment can improve word timestamps after transcription,
but it is less attractive for low-latency streaming. It adds another model,
language-specific alignment constraints, and extra latency. It is better as a
post-processing quality path than as the first live word-streaming mechanism.

### Backend-Native Callbacks

`whisper.cpp` exposes streaming examples and segment callbacks, which can be
useful for low-resource devices. However, ivrit's current wrapper notes that the
basic API does not provide word-level timestamps in a way that fits `Word`
reliably. Segment callbacks alone are not enough for a word-level UI.

## Implementation Plan

1. Extend `WhisperSession` with hypothesis state:
   - previous candidate words,
   - consecutive match counts,
   - emitted word count or emitted audio time,
   - pending stable words since the last `append()`.
2. Decode with `output_options={"word_timestamps": True}` for
   `faster-whisper` and `stable-whisper`.
3. Flatten each decode into word candidates using the existing `Segment.words`
   data.
4. Compute the stable prefix and append newly stable words to a queue.
5. Trim the PCM buffer through the last emitted word minus an overlap window.
6. On `flush()`, emit all remaining words with `is_final=True`.
7. Add synthetic unit tests around the stability policy without loading Whisper
   models. Integration tests can remain optional because model downloads are
   heavy.

## Recommended Default

Use repeated-buffer transcription with `faster-whisper` as the first supported
word-level streaming backend. It fits the current architecture, works without a
backend-specific incremental decoder, and can be tested with pure-Python
stability-policy fixtures.

Use `stable-whisper` as a quality-oriented second backend where the added cost
is acceptable. Treat `whisper.cpp` as segment-streaming only until the wrapper
can reliably expose word timestamps.

## Risks

- Re-decoding a rolling buffer is CPU/GPU expensive. A debounce interval and
  minimum buffer duration are required.
- Aggressive trimming can remove context and make the next decode worse. Keep
  0.25-0.5 seconds of overlap after emitted words.
- Hebrew tokenization may make "word" boundaries less visually intuitive than
  English. UI consumers should render backend words but avoid assuming they map
  perfectly to whitespace-separated text.
- Stability thresholds need real audio calibration before being treated as
  product defaults.

## References

- [`SYSTRAN/faster-whisper`](https://github.com/SYSTRAN/faster-whisper):
  `word_timestamps=True` extracts word-level timestamps using cross-attention
  and dynamic time warping.
- [`ufal/whisper_streaming`](https://github.com/ufal/whisper_streaming):
  local-agreement streaming over repeated Whisper hypotheses.
- Machacek, Dabre, and Bojar,
  ["Turning Whisper into Real-Time Transcription System"](https://aclanthology.org/2023.ijcnlp-demo.3/):
  describes Whisper-Streaming and reports 3.3 second average latency on an
  English long-form ASR benchmark.
- [`ggml-org/whisper.cpp` stream example](https://github.com/ggml-org/whisper.cpp/tree/master/examples/stream):
  demonstrates repeated microphone-window transcription and VAD-driven sliding
  window mode, but ivrit's current Python wrapper should still treat it as
  segment-first.
- [`jianfch/stable-ts`](https://github.com/jianfch/stable-ts): improves Whisper
  timestamp reliability and is best treated as a quality-oriented path rather
  than the first low-latency mechanism.
- [`m-bain/whisperX`](https://github.com/m-bain/whisperX): strong word-level
  alignment reference, best used as a quality post-processing option rather than
  the first low-latency path.
