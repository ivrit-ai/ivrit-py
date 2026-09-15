"""Streaming-stability probe for ivrit-py issue #12.

Simulates the WhisperSession "re-decode whole buffer, commit all-but-last"
loop on a fixed audio file and measures what actually changes between
consecutive decode rounds.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
import difflib
import numpy as np
from faster_whisper import WhisperModel, decode_audio

SR = 16000
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # seconds between decode rounds


def decode_round(model, audio_upto_s):
    t0 = time.perf_counter()
    segs, info = model.transcribe(
        audio_upto_s,
        word_timestamps=True,
        beam_size=5,
        vad_filter=False,
    )
    elapsed = time.perf_counter() - t0
    out = []
    for s in segs:
        out.append({
            "text": s.text,
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "avg_logprob": round(s.avg_logprob, 3),
            "no_speech_prob": round(s.no_speech_prob, 3),
            "words": [
                {"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2),
                 "p": round(w.probability, 3) if w.probability is not None else None}
                for w in (s.words or [])
            ],
        })
    return out, elapsed, getattr(info, "language", None)


def wordseq(segments):
    return " ".join(s["text"].strip() for s in segments)


def main(path, model_name):
    audio = decode_audio(path, sampling_rate=SR)
    dur = len(audio) / SR
    print(f"file={path} duration={dur:.1f}s model={model_name}")
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    cuts = list(range(STEP, int(dur) + 1, STEP))
    rounds = []
    for t in cuts:
        buf = audio[: int(t * SR)]
        segs, elapsed, _lang = decode_round(model, buf)
        rounds.append({"t": t, "decode_s": round(elapsed, 2), "segments": segs})
        print(f"\n--- round t={t}s (decode {elapsed:.2f}s, {len(segs)} segs) ---")
        for i, s in enumerate(segs):
            tag = "COMMIT" if i < len(segs) - 1 else "PENDING"
            print(f"  [{tag}] {s['start']:5.2f}-{s['end']:5.2f} "
                  f"logp={s['avg_logprob']:.2f} nosp={s['no_speech_prob']:.2f} | {s['text']}")

    # --- measurements ---
    print("\n========== STABILITY ANALYSIS ==========")
    # 1) do committed segments stay identical in later rounds?
    leaks = 0
    for r in range(len(rounds) - 1):
        cur, nxt = rounds[r], rounds[r + 1]
        for i, seg in enumerate(cur["segments"][:-1]):  # committed at round r
            if i < len(nxt["segments"]):
                nt = nxt["segments"][i]["text"].strip()
                if nt != seg["text"].strip():
                    leaks += 1
                    print(f"COMMIT-LEAK r{t2s(cur)} seg{i}: '{seg['text'].strip()}' -> '{nt}'")
    print(f"committed-segment changes observed in next round: {leaks}")

    # 2) pending (last) segment text churn between rounds
    for r in range(len(rounds) - 1):
        a = wordseq(rounds[r]["segments"][-1:])
        b = wordseq(rounds[r + 1]["segments"][-1:])
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        print(f"last-seg text similarity t={rounds[r]['t']}->{rounds[r+1]['t']}: {ratio:.2f} "
              f"('{a.strip()[:60]}' -> '{b.strip()[:60]}')")

    # 3) word-timestamp drift for words present in consecutive rounds
    drift = []
    for r in range(len(rounds) - 1):
        w_a = {w["w"]: w for s in rounds[r]["segments"] for w in s["words"]}
        w_b = {w["w"]: w for s in rounds[r + 1]["segments"] for w in s["words"]}
        for w in set(w_a) & set(w_b):
            drift.append(abs(w_a[w]["s"] - w_b[w]["s"]))
    if drift:
        print(f"word-start drift: mean={np.mean(drift):.3f}s max={max(drift):.3f}s n={len(drift)}")

    # 4) compute scaling
    ts = [r["t"] for r in rounds]
    ds = [r["decode_s"] for r in rounds]
    print(f"decode time vs buffer: {list(zip(ts, ds))}")

    with open("probe_results.json", "w", encoding="utf-8") as f:
        json.dump(rounds, f, ensure_ascii=False, indent=1)


def t2s(r):
    return r["t"]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ivrit-py/tests/test_input_10s.mp3",
         sys.argv[2] if len(sys.argv) > 2 else "tiny")
