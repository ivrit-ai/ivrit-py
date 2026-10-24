# Transcription correction

Optional text-to-text post-processor for `faster-whisper` hypotheses.
Default path is passthrough (`engine="off"`). Nothing is downloaded or
loaded unless a caller opts in.

## Constraints (issue #14)

- Speed: wall time on hypotheses only must be **>10× faster** than
  `faster-whisper` on the same CPU, single thread, batch size 1.
  Report tokens/sec as well as seconds.
- License: weights, training code, and data provenance must be
  CC-BY-4.0 / ivrit.ai compatible. No unlabeled web scrapes.
- Integration: opt-in, lazy imports, works offline once weights are
  on disk (`IVRIT_CORRECTOR_WEIGHTS`).

## API

```python
from ivrit.corrector import correct_transcripts

clean = correct_transcripts(["shalom olam"], engine="off")[0]
```

| `engine`         | Behavior                                      |
|------------------|-----------------------------------------------|
| `off`            | Copy input list. No extra deps.               |
| `t5-small-int8`  | Int8 encoder-decoder on local CC-BY-4.0 weights. |

Unknown `engine` values raise `ValueError`. Missing extras raise
`ImportError` with the install line. Missing weights raise
`FileNotFoundError` pointing at `IVRIT_CORRECTOR_WEIGHTS`.

## Benchmark protocol

1. Transcribe `samples/hebrew_eval_16k/*.wav` with `faster-whisper` on
   CPU. Record wall time `T_whisper` and hypothesis strings.
2. Run the corrector on those strings only (no audio). Record
   `T_correct`.
3. Accept when `T_whisper / T_correct > 10` **and**
   `WER_corrected <= WER_raw` on that eval set.
4. Publish CPU, thread count, weight hash, and
   `onnxruntime` / `ctranslate2` versions next to the numbers.

`scripts/bench_corrector.py` times the text path. Pair it with a
separate `faster-whisper` timing run; do not mix audio decode into
`T_correct`.

## Weight guidance

- Encoder-decoder at roughly T5-small scale (~60M params), exported
  int8 (ONNX or CTranslate2).
- Training pairs: `whisper_hypothesis -> gold_transcript` from ivrit.ai
  corpora plus synthetic noise with recorded provenance.
- Ship a `model_card.md` beside the weights: dataset hashes, WER/CER
  deltas, latency table, license.

Until those weights are published, `engine="t5-small-int8"` is a
deliberate hard failure rather than a silent passthrough.
