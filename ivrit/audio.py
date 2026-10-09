import logging
from typing import Generator, Union, List, Optional
from faster_whisper import WhisperModel
from .types import Segment, Word
from .utils import check_dependencies

class TranscriptionModel:
    def transcribe(self, **kwargs):
        raise NotImplementedError

class FasterWhisperModel(TranscriptionModel):
    def __init__(self, model: str, device: str = "auto", model_path: Optional[str] = None):
        check_dependencies(["faster_whisper"], "faster-whisper")
        self.model = WhisperModel(model_path or model, device=device)

    def transcribe(self, path: str, stream: bool = False, **kwargs) -> Union[List[dict], Generator[Segment, None, None]]:
        segments, info = self.model.transcribe(path, **kwargs)
        
        def generator():
            for segment in segments:
                words = [Word(word=w.word, start=w.start, end=w.end, probability=w.probability) for w in segment.words]
                yield Segment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    words=words,
                    extra_data={"confidence": segment.avg_logprob}
                )
        
        if stream:
            return generator()
        return [s for s in generator()]
