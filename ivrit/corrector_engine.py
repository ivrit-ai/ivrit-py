"""Int8 backend for transcription correction.

Imports for the runtime extra stay inside correct_batch so
`engine="off"` never needs them. Weights stay local; this module
does not fetch from the network.
"""

from __future__ import annotations

import os

ENGINE_ID = "ivrit-ai/hebrew-transcript-corrector-t5-small-int8"
LICENSE = "CC-BY-4.0"
ENV_WEIGHTS = "IVRIT_CORRECTOR_WEIGHTS"


def correct_batch(texts: list[str], max_length: int = 256) -> list[str]:
    try:
        import onnxruntime as ort  # optional extra
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for engine='t5-small-int8'; "
            "install with: pip install 'ivrit[corrector]'"
        ) from exc

    weights = os.environ.get(ENV_WEIGHTS, "").strip()
    if not weights or not os.path.isdir(weights):
        raise FileNotFoundError(
            f"set {ENV_WEIGHTS} to a local {LICENSE} weights directory "
            f"for {ENGINE_ID}; see docs/transcription-correction.md"
        )

    # Touch the runtime so a broken extra fails here, not later.
    _ = ort.get_available_providers()
    _ = max_length
    _ = texts

    raise NotImplementedError(
        f"int8 inference for {ENGINE_ID} is not wired; "
        f"weights dir {weights!r} ({LICENSE}). "
        "See docs/transcription-correction.md."
    )
