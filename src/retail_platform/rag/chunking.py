"""Split markdown policies into one chunk per second-level heading."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc-stem>#<section-slug>"
    doc: str
    title: str
    section: str
    text: str
    trusted: bool

    @property
    def search_text(self) -> str:
        return f"{self.title} - {self.section}\n{self.text}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def chunk_markdown(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    trusted = "untrusted" not in title.lower()
    chunks = []
    for match in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", raw, flags=re.M | re.S):
        section, body = match.group(1).strip(), match.group(2).strip()
        chunks.append(Chunk(f"{path.stem}#{slug(section)}", path.stem, title, section, body, trusted))
    return chunks


def load_corpus(directory: Path) -> list[Chunk]:
    return [c for p in sorted(directory.glob("*.md")) for c in chunk_markdown(p)]
