from parsers.text_chunker import chunk_text


def test_chunk_text_basic():
    text = "Слово " * 100
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_overlap():
    text = "A" * 100 + "B" * 100 + "C" * 100
    chunks = chunk_text(text, chunk_size=120, overlap=30)

    assert chunks[1][:30] in chunks[0]


def test_chunk_text_short_text():
    text = "Короткий текст."
    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    chunks = chunk_text("", chunk_size=100, overlap=10)

    assert chunks == []


def test_chunk_text_exact_size():
    text = "X" * 300
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) == 2
    assert chunks[0] == text
    assert chunks[1] == text[-50:]


def test_chunk_text_default_parameters():
    text = "Y" * 5000
    chunks = chunk_text(text)

    assert len(chunks) >= 2
    assert all(len(c) <= 3000 for c in chunks)
