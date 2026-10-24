"""Optional text-to-text transcription corrector (issue #14)."""

from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_ENGINES = ("off", "t5-small-int8")


@dataclass(frozen=True)
class CorrectorConfig:
    engine: str = "off"
    max_length: int = 256


def correct_transcripts(
    texts: list[str],
    engine: str = "off",
    max_length: int = 256,
) -> list[str]:
    """Correct ASR hypotheses.

    engine="off": copy the input (default, no extra packages).
    engine="t5-small-int8": run the optional int8 backend. Raises
    ImportError if extras are missing, FileNotFoundError if weights
    are not on disk, NotImplementedError until inference is wired.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list of str")
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    cfg = CorrectorConfig(engine=engine, max_length=max_length)
    if cfg.engine == "off":
        return list(texts)
    if cfg.engine == "t5-small-int8":
        from ivrit.corrector_engine import correct_batch

        return correct_batch(texts, max_length=cfg.max_length)
    raise ValueError(
        f"unknown corrector engine {cfg.engine!r}; "
        f"expected one of {SUPPORTED_ENGINES}"
    )
