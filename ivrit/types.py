from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import json


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass
class Segment(Serializable):
    """Represents a transcription segment"""

    text: str
    start: float
    end: float
    speakers: List[str] = field(default_factory=list)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class Word(Serializable):
    """Represents a word in a transcription segment"""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None
