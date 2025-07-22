from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Segment:
    """Represents a transcription segment"""

    text: str
    start: float
    end: float
    extra_data: Dict[str, Any]
    speaker: Optional[str] = None

@dataclass
class Word:
    """Represents a word in a transcription segment"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
