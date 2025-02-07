import numpy as np
import pytest

from late_chunking import late_chunking as lc


@pytest.fixture
def documents() -> list[str]:
    return [
        "Berlin is the capital of Germany",
        "A 'quatre-quart' cake consists of flour, butter, sugar and eggs",
        "Ultimate Frisbee is a team sport played with a flying disc",
    ]


@pytest.fixture
def query() -> str:
    return "What is the name of the capital of Germany?"


def test_mini_embedder(documents, query):
    embedder = lc.SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    doc_embs = embedder.embed_documents(documents)
    assert doc_embs.shape == (len(documents), 384)

    q_embs = embedder.embed_query(query)
    assert q_embs.shape == (384,)

    similarities = embedder.model.similarity(q_embs, doc_embs).numpy()
    assert np.argmax(similarities) == 0


def test_token_embeddings(documents, query):
    embedder = lc.SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    doc_embs = embedder.embed_documents(documents, output_value="token_embeddings")
    assert len(doc_embs) == len(documents)
    for emb in doc_embs:
        assert emb.shape[0] > 1
        assert emb.shape[1] == 384

    q_embs = embedder.embed_query(query, output_value="token_embeddings")
    assert q_embs.shape[0] > 1
    assert q_embs.shape[1] == 384


def test_nomic_embedder(documents, query):
    embedder = lc.SentenceTransformerEmbedder(
        "nomic-ai/modernbert-embed-base",
        document_kwargs={"prompt": "search_document: "},
        query_kwargs={"prompt": "search_query: "},
    )

    doc_embs = embedder.embed_documents(documents)
    assert doc_embs.shape == (len(documents), 768)

    q_embs = embedder.embed_query(query)
    assert q_embs.shape == (768,)

    similarities = embedder.model.similarity(q_embs, doc_embs).numpy()
    assert np.argmax(similarities) == 0
