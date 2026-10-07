"""Time the text-to-text corrector against a recorded faster-whisper run.

Protocol (docs/transcription-correction.md):
  T_whisper  — CPU transcription wall time for the eval wavs
  T_correct  — corrector wall time on those hypothesis strings only
  require    — T_whisper / T_correct > 10 and WER does not get worse

This script only measures the text path. Pass --whisper-seconds from a
separate faster-whisper timing run; do not fold audio decode into T_correct.
"""

from __future__ import annotations

import argparse
import json
import time

from ivrit.corrector import correct_transcripts


def time_corrector(texts: list[str], engine: str) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    out = correct_transcripts(texts, engine=engine)
    return out, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        default="off",
        help="corrector engine (default: off)",
    )
    parser.add_argument(
        "--whisper-seconds",
        type=float,
        default=None,
        help="T_whisper from a separate CPU transcription run",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the corrector call (warmup not counted if >1)",
    )
    args = parser.parse_args()

    sample = ["ze haya nisyon transcription", "shalom olam"]
    if args.repeat > 1:
        time_corrector(sample, args.engine)

    out, t_correct = time_corrector(sample, args.engine)
    payload = {
        "n": len(sample),
        "engine": args.engine,
        "T_correct": t_correct,
        "T_whisper": args.whisper_seconds,
        "ratio": (
            args.whisper_seconds / t_correct
            if args.whisper_seconds and t_correct > 0
            else None
        ),
        "out": out,
        "pass_speed": (
            args.whisper_seconds / t_correct > 10
            if args.whisper_seconds and t_correct > 0
            else None
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["pass_speed"] is False:
        raise SystemExit("speed requirement failed: T_whisper / T_correct <= 10")


if __name__ == "__main__":
    main()
