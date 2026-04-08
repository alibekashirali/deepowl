from dataclasses import dataclass
import tiktoken

CHUNK_SIZE = 500     # tokens
CHUNK_OVERLAP = 50   # tokens

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int
    token_count: int


def chunk_document(content: str, source: str) -> list[Chunk]:
    tokens = _enc.encode(content)
    if not tokens:
        return []

    chunks = []
    start = 0
    idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunks.append(Chunk(
            content=chunk_text,
            source=source,
            chunk_index=idx,
            token_count=len(chunk_tokens),
        ))
        idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
