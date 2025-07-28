from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Segment:
    """Represents a transcription segment"""

    text: str
    start: float
    end: float
    speakers: List[str] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class Word:
    """Represents a word in a transcription segment"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
