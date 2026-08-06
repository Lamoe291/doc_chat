from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class Document:
    source: Path
    page_number: int
    text: str
