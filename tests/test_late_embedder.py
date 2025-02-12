import pytest
import torch
import numpy as np
from late_chunking.late_chunking import LateEmbedder

@pytest.fixture
def embedder():
    return LateEmbedder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_seq_length=512,
        max_seq_overlap=50,
        document_prefix="Document: ",
        query_prefix="Query: "
    )

def test_token_length_calculation(embedder):
    prefix_length, leading_special, trailing_special = embedder._get_token_lengths()
    
    # Check prefix length (for "Document: ")
    assert prefix_length == 2  # "Document:" and " " typically tokenize to 2 tokens
    
    # Most models like MiniLM use [CLS] at start and [SEP] at end
    assert leading_special == 1
    assert trailing_special == 1

def test_short_text_embedding(embedder):
    text = "This is a short test text."
    chunks = [(0, len(text))]
    
    embeddings = embedder.calculate_late_embeddings(text, chunks)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 1  # One chunk
    assert embeddings.shape[1] == 384  # MiniLM output dimension

def test_long_text_embedding(embedder):
    # Create text longer than max_seq_length
    text = "word " * 200  # Will create a long text
    chunks = [(0, 100), (100, 200), (200, len(text))]  # Multiple chunks
    
    embeddings = embedder.calculate_late_embeddings(text, chunks)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 384  # MiniLM output dimension

def test_chunk_mapping(embedder):
    text = "This is a test text for mapping chunks to tokens."
    # Get token offsets
    tokens = embedder.embedder.tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False
    )
    token_offsets = tokens["offset_mapping"].squeeze()
    
    # Test mapping of text chunks to tokens
    chunks = [(0, 4), (5, 7)]  # "This", "is"
    token_chunks = embedder._map_chunks_to_tokens(chunks, token_offsets)
    
    assert len(token_chunks) == len(chunks)
    assert all(isinstance(span, tuple) and len(span) == 2 for span in token_chunks)
    assert all(start < end for start, end in token_chunks)

def test_overlapping_chunks(embedder):
    # Test handling of overlapping regions in long texts
    text = "word " * 200
    token_embeddings, token_offsets = embedder._get_long_token_embeddings(text)
    
    assert isinstance(token_embeddings, torch.Tensor)
    assert token_embeddings.dim() == 2
    assert token_embeddings.shape[1] == 384  # embedding dimension

def test_empty_text(embedder):
    text = ""
    chunks = [(0, 0)]
    
    with pytest.raises(Exception):  # Should handle empty text appropriately
        embedder.calculate_late_embeddings(text, chunks)

def test_invalid_chunks(embedder):
    text = "This is a test."
    invalid_chunks = [(10, 5)]  # End before start
    
    with pytest.warns(UserWarning):  # Should warn about invalid chunks
        embedder.calculate_late_embeddings(text, invalid_chunks)
