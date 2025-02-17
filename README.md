# Late Chunking for Text Embeddings

A Python library implementing Late Chunking for improved text embeddings, based on the research by Jina AI presented in their paper ["Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"](https://arxiv.org/html/2409.04701v2).

## What is Late Chunking?

Late Chunking is an embedding method that preserves long-context information within chunk embeddings. Unlike traditional chunking that splits text before embedding, Late Chunking:

1. First embeds the entire text to maintain full context
2. Then splits the token-level embeddings into chunks
3. Finally pools these contextually-rich embeddings into chunk-level representations

This approach helps maintain semantic coherence across chunks while still allowing for manageable chunk sizes.

> To get the most out of late chunking, it is recommended to use an embedding model with a long context length!


## Quick Start

`late_chunking` is written to work with any SentenceTransformer model.

```python
from late_chunking import LateEmbedder

# Initialize the embedder
embedder = LateEmbedder(
    model_name="jinaai/jina-embeddings-v2-base-en",
    model_kwargs={"trust_remote_code": True},
)

# Your text and chunk boundaries
text = "Your long document text here..."
chunks = [(0, 100), (90, 200)]  # Chunk boundaries from any external chunking method

# Calculate embeddings
embeddings = embedder.calculate_late_embeddings(text, chunks)
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

