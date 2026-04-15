from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source_path: Optional[str] = None
    section: Optional[str] = None
    start_offset: Optional[int] = None
    end_offset: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
