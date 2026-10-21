"""
MaintenAI— Industrial Maintenance Manual Markdown & Section Chunker

Splits equipment service manuals, SOPs, and troubleshooting guides into grounded chunks
with preserved structural hierarchy, equipment tags, and failure code metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DocumentChunk:
    """A semantic chunk of a maintenance document with grounded metadata."""

    chunk_id: str
    content: str
    source_file: str
    equipment_type: str
    section_title: str
    breadcrumbs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "equipment_type": self.equipment_type,
            "section_title": self.section_title,
            "breadcrumbs": self.breadcrumbs,
            "metadata": self.metadata,
        }


class MarkdownManualChunker:
    """
    Parses Markdown maintenance manuals and creates context-preserving chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self,
        content: str,
        source_file: str,
        equipment_type: str = "general",
        default_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """
        Split markdown text into grounded chunks by header hierarchy and size.
        """
        chunks: list[DocumentChunk] = []
        lines = content.splitlines()

        current_headers: dict[int, str] = {}
        current_section_lines: list[str] = []
        current_section_title = "Introduction"

        def flush_section() -> None:
            if not current_section_lines:
                return

            section_text = "\n".join(current_section_lines).strip()
            if not section_text:
                return

            breadcrumbs = [current_headers[lvl] for lvl in sorted(current_headers.keys())]

            # Split large sections into overlapping sub-chunks if needed
            words = section_text.split()
            if len(words) <= self.chunk_size:
                sub_texts = [section_text]
            else:
                sub_texts = []
                step = self.chunk_size - self.chunk_overlap
                for i in range(0, len(words), step):
                    sub_words = words[i : i + self.chunk_size]
                    sub_texts.append(" ".join(sub_words))

            for idx, text in enumerate(sub_texts):
                chunk_hash = hashlib.md5(
                    f"{source_file}:{current_section_title}:{idx}:{text[:30]}".encode(),
                    usedforsecurity=False,
                ).hexdigest()[:12]
                chunk_id = f"{Path(source_file).stem}_{chunk_hash}"

                meta = dict(default_metadata or {})
                meta["word_count"] = len(text.split())
                meta["section_index"] = idx

                # Extract any error codes / failure codes mentioned (e.g. HDF, PWF, OSF, TWF, E-102)
                codes = re.findall(r"\b(?:TWF|HDF|PWF|OSF|RNF|ERR-\d+|CODE-[A-Z0-9]+)\b", text)
                if codes:
                    meta["failure_codes"] = list(set(codes))

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=text,
                        source_file=source_file,
                        equipment_type=equipment_type,
                        section_title=current_section_title,
                        breadcrumbs=breadcrumbs.copy(),
                        metadata=meta,
                    )
                )

        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

        for line in lines:
            match = header_pattern.match(line)
            if match:
                flush_section()
                current_section_lines = []
                level = len(match.group(1))
                title = match.group(2).strip()

                # Clean deeper headers
                current_headers = {lvl: h for lvl, h in current_headers.items() if lvl < level}
                current_headers[level] = title
                current_section_title = title
            else:
                current_section_lines.append(line)

        flush_section()
        return chunks
