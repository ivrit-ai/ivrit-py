from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Segment:
    """Represents a transcription segment"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class Word:
    """Represents a word in a transcription segment"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
