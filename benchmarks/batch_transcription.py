#!/usr/bin/env python3
"""
Benchmark the faster-whisper batched transcription path against the plain path.

Examples
--------
Single long file, default settings (tiny model, CPU)::

    python benchmarks/batch_transcription.py --audio tests/asimov.mp3 --model tiny

Match a production-like decoder setup::

    python benchmarks/batch_transcription.py --audio tests/asimov.mp3 \
        --model ivrit-ai/whisper-large-v3-turbo-ct2 --language he --batch-size 16

Many short files (cross-file concurrency, needs a multi-worker model)::

    python benchmarks/batch_transcription.py --audio tests/test_input_10s.mp3 \
        --model tiny --files 8 --batch-workers 4

Accuracy parity check (same 20 s windows through both paths)::

    python benchmarks/batch_transcription.py --audio tests/asimov.mp3 \
        --model tiny --language he --parity

``batch_size`` trades latency for throughput: it decodes several chunks of the
*same* file in one forward pass. ``batch_workers`` is orthogonal - it runs
several *files* concurrently via threads, which only helps when the model was
loaded with ``num_workers`` > 1 and the machine has spare cores / GPU headroom.

The script reports wall time, real-time factor (RTF), speed-up and, when
``--parity`` is used, the word error rate (WER) between the plain and batched
transcripts of identical audio windows. Batched decoding can legitimately
change full-file output because chunk boundaries and cross-chunk conditioning
differ; the parity mode removes both variables, so any difference there is a
real accuracy regression.
"""

from __future__ import annotations

import argparse
import difflib
import os
import statistics
import tempfile
import time
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

from ivrit.audio import FasterWhisperModel
from ivrit.utils import load_audio

SAMPLE_RATE = 16_000


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance, normalised by the reference length."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    distance = sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )
    return distance / len(ref)


def write_wav(chunk: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a float32 mono chunk to a temporary 16-bit WAV file."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    samples = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return path


def measure(model, audio_path, options: Dict, repeats: int) -> Tuple[float, str]:
    """Return the median wall time and the (stable) transcript."""
    texts: List[str] = []
    times: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = model.transcribe(path=audio_path, stream=False, **options)
        times.append(time.perf_counter() - started)
        texts.append(result["text"].strip())
    if len(set(texts)) > 1:
        print("  warning: repeats produced different transcripts (temperature fallback)")
    if repeats > 1:
        print(f"  runs: {[round(t, 2) for t in times]}")
    return statistics.median(times), texts[0]


def measure_batch(model, sources: List[str], options: Dict, repeats: int) -> Tuple[float, str]:
    """Median wall time and joined transcript for a list-of-files call."""
    texts: List[str] = []
    times: List[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        results = model.transcribe(path=sources, stream=False, **options)
        times.append(time.perf_counter() - started)
        texts.append(" ".join(r.get("text", "").strip() for r in results))
    if len(set(texts)) > 1:
        print("  warning: repeats produced different transcripts (temperature fallback)")
    if repeats > 1:
        print(f"  runs: {[round(t, 2) for t in times]}")
    return statistics.median(times), texts[0]


def run_parity(model, audio: np.ndarray, base_options: Dict, batch_size: int, window: float) -> None:
    """Compare plain and batched output on identical fixed windows."""
    print(f"\nAccuracy parity ({window:.0f}s windows, identical chunk boundaries)")
    print("=================================================================")
    options = dict(base_options)
    # Knock out every source of run-to-run variation so the only difference
    # between the two runs is batched vs unbatched decoding.
    options.update({"vad_filter": False, "condition_on_previous_text": False, "temperature": 0.0})

    references: List[str] = []
    hypotheses: List[str] = []
    step = int(window * SAMPLE_RATE)
    for start in range(0, len(audio) - SAMPLE_RATE, step):
        chunk = audio[start : start + step]
        path = write_wav(chunk)
        try:
            plain = model.transcribe(path=path, stream=False, **options)
            batched = model.transcribe(
                path=path,
                stream=False,
                batch_size=batch_size,
                clip_timestamps=[{"start": 0, "end": len(chunk) / SAMPLE_RATE}],
                **options,
            )
        finally:
            os.remove(path)
        wer = word_error_rate(plain["text"], batched["text"])
        references.append(plain["text"])
        hypotheses.append(batched["text"])
        print(
            f"  {start / SAMPLE_RATE:6.1f}s  "
            f"plain={len(plain['text']):>4}c  batched={len(batched['text']):>4}c  "
            f"WER={wer * 100:5.1f}%"
        )
    print(f"  overall WER: {word_error_rate(' '.join(references), ' '.join(hypotheses)) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", required=True, help="Audio file used for the benchmark")
    parser.add_argument("--model", default="tiny", help="faster-whisper model size or path")
    parser.add_argument("--language", default=None, help="Language code (default: auto-detect)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="Chunks per batched pass (default: 8)")
    parser.add_argument("--files", type=int, default=1, help="Repeat the audio N times to emulate a multi-file batch")
    parser.add_argument("--batch-workers", type=int, default=1, help="Files decoded concurrently (default: 1)")
    parser.add_argument("--repeat", type=int, default=1, help="Timed repetitions per configuration (default: 1)")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    parser.add_argument("--compute-type", default=None, help="e.g. float16, int8 (default: engine choice)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CTranslate2 intra-op threads (default: engine choice)")
    parser.add_argument("--parity", action="store_true", help="Also run the fixed-window accuracy parity check")
    parser.add_argument("--parity-window", type=float, default=20.0, help="Parity window length in seconds (default: 20)")
    args = parser.parse_args()

    model_kwargs = {}
    if args.compute_type:
        model_kwargs["compute_type"] = args.compute_type
    if args.cpu_threads:
        model_kwargs["cpu_threads"] = args.cpu_threads
    if args.batch_workers > 1:
        # CTranslate2 only runs concurrent transcribe() calls in parallel when
        # the model is loaded with several workers.
        model_kwargs["num_workers"] = args.batch_workers

    print(f"Loading model {args.model!r} on {args.device or 'auto'} ...")
    model = FasterWhisperModel(model=args.model, device=args.device, **model_kwargs)

    audio = load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE

    base_options: Dict = {
        "language": args.language,
        "beam_size": args.beam_size,
        "word_timestamps": True,
        # The batched pipeline requires VAD (or clip_timestamps) for audio
        # longer than 30 s. Ask for it on both paths so the comparison is fair.
        "vad_filter": True,
        "without_timestamps": False,
    }

    print(f"\nAudio: {args.audio} ({duration:.2f}s), files={args.files}")

    plain_times: List[float] = []
    plain_text = ""
    for _ in range(args.files):
        elapsed, text = measure(model, args.audio, base_options, args.repeat)
        plain_times.append(elapsed)
        plain_text = text
    plain_total = sum(plain_times)

    if args.files > 1 and args.batch_workers > 1:
        batched_options = dict(base_options, batch_workers=args.batch_workers)
        batched_total, batched_text = measure_batch(
            model, [args.audio] * args.files, batched_options, args.repeat
        )
        label = f"files={args.files} workers={args.batch_workers}"
    else:
        batched_options = dict(base_options, batch_size=args.batch_size)
        batched_total, batched_text = measure(
            model, args.audio, batched_options, args.repeat
        )
        label = f"batch_size={args.batch_size}"

    print("\nThroughput")
    print("==========")
    print(f"plain           : {plain_total:8.2f}s  RTF={plain_total / (duration * args.files):.3f}")
    print(f"{label:<15} : {batched_total:8.2f}s  RTF={batched_total / (duration * args.files):.3f}")
    print(f"speed-up        : {plain_total / batched_total:.2f}x")
    print(f"full-file WER (batched vs plain): {word_error_rate(plain_text, batched_text) * 100:.2f}%")
    print("note: full-file WER is informational - chunk boundaries and cross-chunk")
    print("      conditioning differ. Use --parity for an apples-to-apples check.")

    if args.parity:
        run_parity(model, audio, base_options, args.batch_size, args.parity_window)


if __name__ == "__main__":
    main()
