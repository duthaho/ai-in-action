"""
Text chunking strategies for RAG pipelines.

Three strategies, each suited to different content types:
- FixedSizeChunker: Simple token-count based splitting
- SentenceChunker: Respects sentence boundaries
- RecursiveChunker: Respects document structure (headers, paragraphs, sentences)
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk of text with metadata about its origin."""
    text: str
    index: int           # position in the original document
    source: str = ""     # document identifier
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FixedSizeChunker:
    """
    Split text into fixed-size chunks by character count.

    Simple and predictable. Best for uniform content where structure
    doesn't matter much (e.g., plain text logs).
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []

        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=index,
                    source=source,
                ))
                index += 1

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks


class SentenceChunker:
    """
    Split text into chunks that respect sentence boundaries.

    Avoids cutting mid-sentence. Each chunk accumulates sentences
    until reaching the target size, then starts a new chunk.
    Best for prose, documentation, and articles.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple heuristics."""
        sentences = []
        current = []

        for char in text:
            current.append(char)
            if char in ".!?" and len(current) > 10:
                sentence = "".join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        # Don't lose trailing text
        remaining = "".join(current).strip()
        if remaining:
            sentences.append(remaining)

        return sentences

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks = []
        current_chunk: list[str] = []
        current_length = 0
        index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds the limit, finalize current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(Chunk(text=chunk_text, index=index, source=source))
                    index += 1

                # Keep overlap: take sentences from the end of current chunk
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)

                current_chunk = overlap_sentences
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, index=index, source=source))

        return chunks


class RecursiveChunker:
    """
    Split text by structural separators, recursively falling back to
    finer-grained separators when chunks are too large.

    Separator hierarchy: double newlines → single newlines → sentences → characters.

    Best for structured documents: markdown, code, technical docs.
    This is the strategy most production RAG systems use.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the separator hierarchy."""
        if not text:
            return []

        # If text fits in one chunk, return it
        if len(text) <= self.chunk_size:
            return [text]

        # No separators left — hard split by character
        if not separators:
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start += self.chunk_size - self.chunk_overlap
            return chunks

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        parts = text.split(separator)

        # Merge small parts into chunks, recurse on parts that are too large
        result = []
        current_parts: list[str] = []
        current_length = 0

        for part in parts:
            part_len = len(part) + len(separator)

            if current_length + part_len > self.chunk_size and current_parts:
                merged = separator.join(current_parts).strip()
                if merged:
                    result.append(merged)

                # Overlap: keep trailing parts
                overlap_parts = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)

                current_parts = overlap_parts
                current_length = overlap_len

            # If a single part is too large, recurse with finer separators
            if len(part) > self.chunk_size:
                # First, flush current buffer
                if current_parts:
                    merged = separator.join(current_parts).strip()
                    if merged:
                        result.append(merged)
                    current_parts = []
                    current_length = 0

                sub_chunks = self._split_text(part, remaining_separators)
                result.extend(sub_chunks)
            else:
                current_parts.append(part)
                current_length += part_len

        # Final buffer
        if current_parts:
            merged = separator.join(current_parts).strip()
            if merged:
                result.append(merged)

        return result

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        if not text.strip():
            return []

        raw_chunks = self._split_text(text, self.separators)

        return [
            Chunk(text=c.strip(), index=i, source=source)
            for i, c in enumerate(raw_chunks)
            if c.strip()
        ]
