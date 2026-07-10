def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
