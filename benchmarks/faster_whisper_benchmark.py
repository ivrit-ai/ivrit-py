#!/usr/bin/env python3
"""Reproducible benchmarks for the faster-whisper engine.

See PERFORMANCE.md for the results this script produced and how to read them.
Run it from the repository root:

    python benchmarks/faster_whisper_benchmark.py stages
    python benchmarks/faster_whisper_benchmark.py ab tests/asimov.mp3
    python benchmarks/faster_whisper_benchmark.py lengths tests/asimov.mp3
    python benchmarks/faster_whisper_benchmark.py session tests/asimov.mp3
    python benchmarks/faster_whisper_benchmark.py batch tests/asimov.mp3

``ab`` and ``lengths`` compare the optimized call with vanilla faster-whisper.
``batch`` measures faster-whisper's batched pipeline for reference; it is not
enabled by this change (see #42).

Every wall-clock number is a median over repeated runs. ``cpu`` is
``time.process_time()`` (all threads of this process), so ``cpu / wall`` shows
how many cores the process keeps busy.
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "ivrit-ai/whisper-large-v3-turbo-ct2"
DEFAULT_AUDIO = os.path.join(REPO_ROOT, "tests", "asimov.mp3")


def _model(args):
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    # Warm up kernels and the allocator so the first measured run is realistic.
    list(model.transcribe(os.path.join(REPO_ROOT, "tests", "test_input_10s.mp3"),
                          language="he", word_timestamps=True)[0])
    return model


def _consume(segments):
    return [(s.start, s.end, s.text,
             tuple((w.word, w.start, w.end) for w in (s.words or [])))
            for s in segments]


def _duration(path):
    """Media duration in seconds, via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def _timed_pair(baseline, candidate, repeats):
    """Time two callables round-robin so clock drift hits both arms equally."""
    base_walls, cand_walls = [], []
    base_out = cand_out = None
    for _ in range(repeats):
        w0 = time.perf_counter()
        base_out = baseline()
        base_walls.append(time.perf_counter() - w0)
        w0 = time.perf_counter()
        cand_out = candidate()
        cand_walls.append(time.perf_counter() - w0)
    return statistics.median(base_walls), statistics.median(cand_walls), base_out, cand_out


# --------------------------------------------------------------------------
# stages: where wall and CPU time actually go
# --------------------------------------------------------------------------
STAGE_NAMES = {
    "decode_audio": "audio decode (PyAV)",
    "log_mel": "log-Mel features",
    "encode": "encoder (GPU)",
    "generate": "decoder + sampling (GPU/CPU)",
    "word_ts": "word alignment",
    "split": "segment splitting",
    "detect_language": "language detection (nested)",
}


def stages(args):
    import faster_whisper.transcribe as fwt
    from faster_whisper import WhisperModel

    totals: dict[str, dict[str, float]] = {}

    @contextmanager
    def stage(name):
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            entry = totals.setdefault(name, {"wall": 0.0, "cpu": 0.0, "n": 0})
            entry["wall"] += time.perf_counter() - w0
            entry["cpu"] += time.process_time() - c0
            entry["n"] += 1

    def wrap(owner, attr, name):
        original = getattr(owner, attr)

        def wrapper(*a, **kw):
            with stage(name):
                return original(*a, **kw)

        setattr(owner, attr, wrapper)

    wrap(fwt, "decode_audio", "decode_audio")
    wrap(fwt.FeatureExtractor, "__call__", "log_mel")
    wrap(fwt.WhisperModel, "encode", "encode")
    wrap(fwt.WhisperModel, "generate_with_fallback", "generate")
    wrap(fwt.WhisperModel, "add_word_timestamps", "word_ts")
    wrap(fwt.WhisperModel, "_split_segments_by_timestamps", "split")
    wrap(fwt.WhisperModel, "detect_language", "detect_language")

    for path in args.audio:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        list(model.transcribe(os.path.join(REPO_ROOT, "tests", "test_input_10s.mp3"),
                              language="he")[0])
        totals.clear()
        walls, cpus = [], []
        for _ in range(args.repeats):
            w0, c0 = time.perf_counter(), time.process_time()
            segments, info = model.transcribe(path, language=None, word_timestamps=True)
            _consume(segments)
            walls.append(time.perf_counter() - w0)
            cpus.append(time.process_time() - c0)

        wall = statistics.median(walls)
        cpu = statistics.median(cpus)
        print(f"\n== {os.path.basename(path)} ==  audio={info.duration:.1f}s "
              f"wall={wall:.2f}s cpu={cpu:.2f}s cpu/wall={cpu / wall:.2f}")
        print(f"{'stage':32s} {'wall':>8s} {'cpu':>8s} {'%wall':>7s}")
        for name, entry in sorted(totals.items(), key=lambda kv: -kv[1]["wall"]):
            label = STAGE_NAMES.get(name, name)
            stage_wall = entry["wall"] / len(walls)
            stage_cpu = entry["cpu"] / len(walls)
            print(f"{label:32s} {stage_wall:8.3f} {stage_cpu:8.3f} "
                  f"{100 * stage_wall / wall:6.1f}%")
        del model


# --------------------------------------------------------------------------
# ab: ivrit default path vs vanilla faster-whisper
# --------------------------------------------------------------------------
def ab(args):
    from ivrit import _fw_perf

    model = _model(args)
    for path in args.audio:
        def vanilla():
            segments, _ = model.transcribe(path, language=None, word_timestamps=True)
            return _consume(segments)

        def optimized():
            segments, _ = _fw_perf.transcribe(
                model, path, language=None, word_timestamps=True)
            return _consume(segments)

        t_v, t_o, out_v, out_o = _timed_pair(vanilla, optimized, args.repeats)
        print(f"{os.path.basename(path):28s} vanilla={t_v:7.3f}s optimized={t_o:7.3f}s "
              f"gain={100 * (t_v - t_o) / t_v:5.1f}% identical={out_v == out_o}")


# --------------------------------------------------------------------------
# lengths: how the gain scales with audio duration
# --------------------------------------------------------------------------
def lengths(args):
    from ivrit import _fw_perf

    model = _model(args)
    source = args.audio[0]
    source_seconds = _duration(source)
    targets = [s for s in (1, 3, 7, 15, 25, 29.5, 30, 30.5, 45, 59, 60, 61, 90, 125, 300)
               if s <= source_seconds]
    print(f"{'seconds':>8s} {'vanilla':>9s} {'optimized':>10s} {'gain':>7s} {'identical':>10s}")
    with tempfile.TemporaryDirectory() as tmp:
        for seconds in targets:
            cut = os.path.join(tmp, f"cut_{seconds}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-t", str(seconds),
                 "-i", source, "-ac", "1", "-ar", "16000", cut],
                check=True)
            vanilla = lambda: _consume(
                model.transcribe(cut, language=None, word_timestamps=True)[0])
            optimized = lambda: _consume(
                _fw_perf.transcribe(model, cut, language=None, word_timestamps=True)[0])
            t_v, t_o, out_v, out_o = _timed_pair(vanilla, optimized, args.repeats)
            print(f"{seconds:8.1f} {t_v:9.3f} {t_o:10.3f} {100 * (t_v - t_o) / t_v:6.1f}% "
                  f"{str(out_v == out_o):>10s}")


# --------------------------------------------------------------------------
# session: incremental PCM path (ivrit's streaming use case)
# --------------------------------------------------------------------------
def session(args):
    import ivrit._fw_perf as fw
    from ivrit import load_model

    real = fw.transcribe

    def vanilla(model, audio, **kw):
        return model.transcribe(audio, **kw)

    def pcm(path):
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
            capture_output=True, check=True).stdout

    data = pcm(args.audio[0])
    chunk = args.chunk_seconds * 32000
    model = load_model(engine="faster-whisper", model=args.model, device=args.device)

    def run(chunk_bytes):
        session_obj = model.create_session(language=None, sample_rate=16000, verbose=False)
        for i in range(0, len(data), chunk_bytes):
            session_obj.append(data[i:i + chunk_bytes])
        session_obj.flush()
        return [(s.start, s.end, s.text) for s in session_obj.get_all_segments()]

    walls = {"vanilla": [], "optimized": []}
    cpus = {"vanilla": [], "optimized": []}
    outs = {}
    try:
        # Alternate the two arms so GPU clock drift affects them equally.
        for _ in range(args.repeats):
            for name, fn in (("vanilla", vanilla), ("optimized", real)):
                fw.transcribe = fn
                w0, c0 = time.perf_counter(), time.process_time()
                outs[name] = run(chunk)
                walls[name].append(time.perf_counter() - w0)
                cpus[name].append(time.process_time() - c0)
    finally:
        fw.transcribe = real

    for name in ("vanilla", "optimized"):
        print(f"{name:10s} wall={statistics.median(walls[name]):6.2f}s "
              f"cpu={statistics.median(cpus[name]):6.2f}s segments={len(outs[name])}")
    v, o = statistics.median(walls["vanilla"]), statistics.median(walls["optimized"])
    print(f"gain={100 * (v - o) / v:.1f}%  identical={outs['vanilla'] == outs['optimized']}")


# --------------------------------------------------------------------------
# batch: measure faster-whisper's batched pipeline (not enabled by this PR)
# --------------------------------------------------------------------------
def batch(args):
    import difflib

    from faster_whisper import BatchedInferencePipeline

    model = _model(args)
    pipeline = BatchedInferencePipeline(model=model)
    for path in args.audio:
        def sequential():
            segments, _ = model.transcribe(path, language=None, word_timestamps=True)
            return _consume(segments)

        def batched():
            segments, _ = pipeline.transcribe(
                path, language=None, word_timestamps=True, batch_size=args.batch_size,
                without_timestamps=False)
            return _consume(segments)

        t_seq, t_bat, out_seq, out_bat = _timed_pair(sequential, batched, args.repeats)
        a = " ".join(s[2] for s in out_seq)
        b = " ".join(s[2] for s in out_bat)
        print(f"{os.path.basename(path):28s} sequential={t_seq:7.3f}s "
              f"batched={t_bat:7.3f}s speedup={t_seq / t_bat:4.2f}x "
              f"segments={len(out_seq)}/{len(out_bat)} "
              f"words={sum(len(s[3]) for s in out_seq)}/{sum(len(s[3]) for s in out_bat)} "
              f"text_similarity={difflib.SequenceMatcher(None, a, b).ratio():.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["stages", "ab", "lengths", "session", "batch"])
    parser.add_argument("audio", nargs="*", default=[DEFAULT_AUDIO])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=int, default=10)
    args = parser.parse_args()
    if not args.audio:
        args.audio = [DEFAULT_AUDIO]

    {"stages": stages, "ab": ab, "lengths": lengths,
     "session": session, "batch": batch}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
