# Word-Level Streaming: Alternatives Analysis

> Analysis for issue #12 — "Analyze existing streaming alternatives using
> whisper, including specifically repeat calls to transcribe with confidence
> analysis."
>
> Method: code review of the current session pipeline plus a reproducible
> measurement probe (`docs/streaming_probe.py`) run against the repo's own
> test fixtures. All numbers below are measured, not estimated.

## 1. What the current session does

`WhisperSession._transcribe_buffered` (`ivrit/audio.py`) implements streaming
as **whole-buffer re-decode on every `append()`**:

1. Buffer raw PCM.
2. Wrap the entire remaining buffer in an in-memory WAV and run
   `model.transcribe_core(blob=...)` over it.
3. If the decode yields more than one segment, emit `all_segments[:-1]` as
   "complete" and trim the buffer to `complete_segments[-1].end`.
4. The last segment's audio stays in the buffer and is re-decoded together
   with newly appended audio on the next call.
5. `flush()` emits whatever remains.

The commit decision is therefore **positional** ("everything but the last
segment is done"), not content-based. No confidence signal participates in
the decision, even though the class docstring describes "confidence
tracking" and the engines already surface the signals (see §3.4).

## 2. Measured behavior of the current policy

Probe: `docs/streaming_probe.py`, `faster-whisper` `tiny`, CPU/int8,
`beam_size=5`, `vad_filter=False`, `word_timestamps=True` (the session
default). Fixtures: `tests/test_input_10s.mp3` (10 s continuous Hebrew
speech) and `tests/asimov.mp3` (125 s multi-segment speech).

### 2.1 Commit starvation on continuous speech

On the 10 s fixture every decode round returned **exactly one segment** —
continuous speech with no internal pause the decoder would split on.
Under all-but-last, `all_segments[:-1]` is empty every round: **nothing
commits until `flush()`**. Latency on un-paused speech is unbounded.

### 2.2 Committed segments are not stable ("commit leak")

On the 125 s fixture, decoded at 20 s intervals simulating appends:

- **55 committed-segment mutations** were observed between consecutive
  rounds — segments emitted as final changed text, and twice the decoder
  re-segmented whole regions (r80→r100 rewrote ~25 segments, boundaries
  included).
- The pending (last) segment's text similarity between consecutive rounds
  was **0.13–0.32** — near-total churn, as designed.
- Word-start drift across rounds: mean **2.80 s**, max **71.9 s**
  (n=325) — timestamps move with re-segmentation, so a downstream consumer
  cannot rely on committed times either.

"Commit" as currently defined is not a stability guarantee; it only means
"the decoder happened to emit a later segment this round."

### 2.3 The tail re-decode dominates cost

Each round re-decodes everything since the last commit. Decode time scales
linearly with buffer length, so a session's total work is quadratic in its
audio length (measured flat ~0.3 s/round on `tiny`+int8; the constant grows
with model size). The all-but-last policy keeps the pending region small,
which masks this on typical inputs, but combined with 2.1 the failure mode
is worst-case on exactly the input where it matters.

### 2.4 Confidence signals do flag bad decodes

The first round (2 s of audio) produced a cross-language hallucination
(Arabic text on Hebrew audio) with `avg_logprob = -1.06` — the worst value
of all rounds, vs −0.45…−0.67 for the later sane decodes. `no_speech_prob`
was unremarkable (0.11). This suggests **`avg_logprob` is a usable commit
gate** and the current positional policy ignores it.

## 3. Design space

Alternatives, ordered by implementation footprint against the current
seams. `transcribe_core` already emits `Segment.words` (word timestamps on
by default) and `_copy_segment_extra_data` already preserves backend
fields such as `avg_logprob` / `no_speech_prob` into `extra_data` — so
options (a)–(d) need no backend changes.

### (a) Confidence-gated commit

Keep the current loop, but commit a non-last segment only when
`avg_logprob > τ` (and optionally `no_speech_prob < τ₂`). Measured basis:
the hallucination round (−1.06) sits clearly below the sane cluster
(−0.45…−0.67), so a threshold near −0.7 separates them on these fixtures.
Fixes the leak class shown in 2.2 and the hallucination class in 2.4.
Cost: a few lines in `_transcribe_buffered`; `extra_data` must be enabled
or the fields surfaced onto `Segment`.

### (b) LocalAgreement-style prefix commit (whisper_streaming policy)

Instead of committing by position, compare **two consecutive decodes** and
commit only the longest common prefix (normalized text) of the confirmed
region, trimming audio at the agreed timestamp. This is the policy used by
`whisper_streaming` (UFAL) / whisper.cpp's stream example and directly
targets 2.2: a segment is emitted only after two independent decodes agree
on it. Higher flicker resistance; one extra decode in flight (already paid
— the session re-decodes anyway).

### (c) Word-level commit granularity

Commit *words* (which carry `start`/`end`/`probability`) rather than whole
segments: emit words up to the last point where consecutive decodes agree,
or up to the last word whose `probability ≥ τ`. Bounds the blast radius of
re-segmentation (2.2 shows segment boundaries are the unstable unit) and
enables mid-segment commit, which also fixes starvation (2.1): even inside
one never-ending segment, agreed words can stream out.

### (d) VAD-segmented decode

Gate commits on silence: run VAD (faster-whisper's `vad_filter`, or
`silero-vad` upstream of the session) and only finalize segments that end
at a detected silence boundary. Speech is re-decoded at utterance level —
the most stable unit, at the cost of coarser commit granularity (per
utterance, not per word) and a VAD dependency.

### (e) Context carry-over (`initial_prompt` / prefix conditioning)

Re-decoding the tail loses left context, which is part of why late text
mutates. Passing the committed transcript as `initial_prompt` (exposed by
faster-whisper) stabilizes style/vocabulary continuity. Known hazard:
prompt-induced repetition loops on long prompts — cap the carried context
(e.g., last ~200 tokens).

### (f) External implementations worth referencing

- **whisper_streaming** (UFAL LocalAgreement-2): reference implementation
  of (b); MIT-licensed, wraps whisper backends — policy portable without
  the dependency.
- **whisper.cpp `stream.cpp`**: sliding-window (typically ~3 s step, 10 s
  window) re-decode with heuristics — demonstrates the rolling-window
  variant that bounds per-decode cost independent of session length.
- **SimulWhisper / AlignAtt**: true simultaneous decoding (attention-guided
  read/write policy). Much better latency/flicker trade-offs, but requires
  model-level access faster-whisper does not expose — out of scope for a
  session-layer change.
- **stable-ts**: exposes segment refinement/alignment; relevant because
  `StableWhisperModel` shares `WhisperSession`, so any session-level fix
  covers both engines.

### (g) Engine coverage constraint

`WhisperSession` serves `faster-whisper` and `stable-whisper`; a
session-layer change fixes both at once. `whisper-cpp` and `runpod`
implement `create_session` as unsupported — remote/foreign-runtime
streaming is a separate problem (server-side streaming or chunked
requests) and should stay out of scope.

## 4. Recommendation

Phase it inside the existing seam rather than replacing the session:

1. **(a)+(b) first**: LocalAgreement-style prefix commit with an
   `avg_logprob` gate in `_transcribe_buffered`. Purely positional →
   evidence-based commit; kills both measured failure classes (leak,
   hallucination commit) with no new dependencies.
2. **(c) next**: word-level commit to fix starvation and shrink the
   re-segmentation blast radius; `Word` already carries `probability` and
   timestamps.
3. **Rolling window cap** (whisper.cpp-style, e.g. re-decode last ~15 s +
   carry committed text via `initial_prompt`) to bound the O(n²) cost for
   long sessions.
4. VAD-gated commits (d) as an optional mode for utterance-boundary
   consumers.

## 5. Appendix: probe

`docs/streaming_probe.py` reproduces every number above:

```bash
pip install faster-whisper
python docs/streaming_probe.py tests/asimov.mp3 tiny 20      # 125 s, 20 s steps
python docs/streaming_probe.py tests/test_input_10s.mp3 tiny 2
```

It simulates the session loop (re-decode a growing buffer), prints
COMMIT/PENDING per segment with `avg_logprob`/`no_speech_prob`, then
reports commit leaks, last-segment similarity, word-timestamp drift, and
per-round decode time. Useful for tuning the τ threshold on real Hebrew
traffic before shipping option (a).

### Caveats

Measurements use `tiny`/int8 on two fixtures — absolute numbers will shift
with model size and audio, but the structural findings (commit leak,
starvation on continuous speech, logprob separating the hallucination
round) are properties of the policy, not of this model.
