from __future__ import annotations

from pathlib import Path

HF_REPO = "notmax123/blue-onnx"
RENIKUD_URL = "https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx"
VOICE_URL = "https://raw.githubusercontent.com/maxmelichov/BlueTTS/refs/heads/main/voices/female1.json"


def _ensure_models() -> tuple[str, str, str]:
    from huggingface_hub import snapshot_download, hf_hub_download
    import requests

    onnx_dir = snapshot_download(
        HF_REPO,
        repo_type="model",
        ignore_patterns=["voices/all_voices/**"],
    )

    renikud_path = Path(onnx_dir) / "renikud.onnx"
    if not renikud_path.exists():
        r = requests.get(RENIKUD_URL)
        r.raise_for_status()
        renikud_path.write_bytes(r.content)

    voice_cache = Path(onnx_dir) / "female1.json"
    if not voice_cache.exists():
        r = requests.get(VOICE_URL)
        r.raise_for_status()
        voice_cache.write_bytes(r.content)

    return onnx_dir, str(voice_cache), str(renikud_path)


class TTS:
    def __init__(
        self,
        onnx_dir: str | None = None,
        style_json: str | None = None,
        renikud_path: str | None = None,
    ):
        try:
            from blue_onnx import BlueTTS
        except ImportError:
            raise ImportError(
                "TTS requires the 'tts' extras: pip install ivrit[tts]"
            )

        if onnx_dir is None or style_json is None or renikud_path is None:
            _onnx_dir, _style_json, _renikud_path = _ensure_models()
            onnx_dir = onnx_dir or _onnx_dir
            style_json = style_json or _style_json
            renikud_path = renikud_path or _renikud_path

        self._tts = BlueTTS(
            onnx_dir=onnx_dir,
            style_json=style_json,
            renikud_path=renikud_path,
        )

    def synthesize(self, text: str, lang: str = "he"):
        return self._tts.synthesize(text, lang=lang)
