from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class EmbeddingWrapper(ABC):
    @abstractmethod
    def embed_documents(self, documents: list[str], **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def embed_query(self, query: str, **kwargs) -> np.ndarray:
        pass


class SentenceTransformerEmbedder(EmbeddingWrapper):
    def __init__(
        self,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        document_kwargs: dict[str, Any] | None = None,
        query_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # TODO: when do I need `output_value='token_embeddings'`?
        self._document_kwargs = document_kwargs if document_kwargs is not None else {}
        self._query_kwargs = query_kwargs if query_kwargs is not None else {}

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.model = SentenceTransformer(model_name, **model_kwargs)

    def embed_documents(self, documents: list[str], **kwargs) -> np.ndarray:
        final_kwargs = self._document_kwargs.copy()
        final_kwargs.update(kwargs)
        return self.model.encode(documents, **final_kwargs)

    def embed_query(self, query: str, **kwargs) -> np.ndarray:
        final_kwargs = self._query_kwargs.copy()
        final_kwargs.update(kwargs)
        return self.model.encode(query, **final_kwargs)


class LateEmbedder:
    """Calculate chunk embeddings via late chunking.

    This method allows to have more context information in chunk embeddings,
    as developed and demonstrated by Jina AI.

    TODO:
    - Jina3, Nomic rely on prefixes for encoding
    - hand in a class that allows to encode query, encode documents
        - simple wrapper to create from HF AutoModel
    """

    def __init__(
        self,
        tokenizer_name: str,
        embedder: EmbeddingWrapper,
        trust_remote_code: bool = True,
        max_seq_length: int = 8192,
        max_seq_overlap: int = 500,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=trust_remote_code
        )
        self.embedding_model = embedder
        self.max_seq_length = max_seq_length
        self.max_Seq_overlap = max_seq_overlap

    def _map_chunks_to_tokens(
        self, chunk_annotations: list[tuple[int, int]], token_offsets: np.ndarray
    ) -> list[tuple[int, int]]:
        """Translate chunk boundaries from text to token space, handling misalignments

        Semantic Segmentation needs to adjust chunk boundaries slightly,
        this can lead to misalignment with tokens.

        Especially when using late chunking, the exact boundaries of chunks in token space
        should not matter that much.
        """
        logger.debug(
            f"Mapping {len(chunk_annotations)} chunks onto {len(token_offsets)} tokens"
        )
        token_chunk_annotations = []

        for chunk_start, chunk_end in chunk_annotations:
            # For start: find token whose span contains or is closest after chunk_start
            start_distances = np.where(
                token_offsets[:, 1] > chunk_start,
                token_offsets[:, 0] - chunk_start,
                np.inf,
            )
            start_token = np.argmin(np.abs(start_distances))

            # For end: find token whose span contains or is closest before chunk_end
            end_distances = np.where(
                token_offsets[:, 0] < chunk_end,
                token_offsets[:, 1] - chunk_end,
                -np.inf,
            )
            end_token = np.argmin(np.abs(end_distances)) + 1  # +1 for exclusive end

            # Ensure we have valid token spans
            if start_token >= end_token:
                logger.warning(
                    f"Invalid token span [{start_token}, {end_token}] for chunk [{chunk_start}, {chunk_end}]"
                )
                end_token = start_token + 1

            token_chunk_annotations.append((start_token, end_token))

        return token_chunk_annotations

    def _get_long_token_embeddings(self, model_inputs):
        """Handle inputs that might be longer than the embedding model can handle.

        Jina uses lower long-chunk sizes than the 8K tokens the model is capable of.
        Also: this method is not tested for multi-sequence inputs!
        """
        task = "retrieval.passage"
        task_id = self.embedding_model._adaptation_map[task]

        max_seq_length = config.conf.select(
            "embeddings.late_chunking.long_seq_length", 2048
        )

        # only single string `input_text`
        adapter_mask = torch.full((1,), task_id, dtype=torch.int32)

        if model_inputs["input_ids"].numel() <= max_seq_length:
            with torch.no_grad():
                model_output = self.embedding_model(
                    **model_inputs, adapter_mask=adapter_mask
                )

            token_embeddings = model_output["last_hidden_state"].squeeze(0).float()

        else:
            # Split tokens into overlapping chunks
            overlap = config.conf.select("index.late_chunking.long_seq_overlap", 256)
            chunks = []
            chunk_embeddings = []

            # Create chunks with overlap
            for i in range(
                0, len(model_inputs["input_ids"].squeeze()), max_seq_length - overlap
            ):
                chunk = {
                    k: v[:, i : i + max_seq_length]
                    if isinstance(v, torch.Tensor)
                    else v
                    for k, v in model_inputs.items()
                }
                chunks.append(chunk)

            # Get embeddings for each chunk
            for chunk in chunks:
                with torch.no_grad():
                    chunk_output = self.embedding_model(
                        **chunk, adapter_mask=adapter_mask
                    )
                chunk_embeddings.append(
                    chunk_output["last_hidden_state"].squeeze(0).float()
                )

            # Combine embeddings from overlapping regions by averaging
            # NOTE: reference implementation of Jina only uses the embeddings of the later chunk
            # https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/mteb_chunked_eval.py#L128
            token_embeddings = torch.zeros(
                (len(model_inputs["input_ids"][0]), chunk_embeddings[0].shape[1]),
                dtype=chunk_embeddings[0].dtype,
            )

            counts = torch.zeros(len(model_inputs["input_ids"][0]), dtype=torch.int)

            pos = 0
            for chunk_emb in chunk_embeddings:
                chunk_size = chunk_emb.shape[0]
                token_embeddings[pos : pos + chunk_size] += chunk_emb
                counts[pos : pos + chunk_size] += 1
                pos += max_seq_length - overlap

            # Average overlapping regions
            token_embeddings = token_embeddings / counts.unsqueeze(1)

        return token_embeddings

    def calculate_late_embeddings(
        self, input_text: str, chunk_annotations: list[tuple[int, int]]
    ) -> list[list[float]]:
        """Calculate late embeddings for chunks in a longer text"""
        task = "retrieval.passage"
        task_prefix = self.embedding_model._task_instructions[task]

        inputs = self.tokenizer(
            task_prefix + input_text, return_tensors="pt", return_offsets_mapping=True
        )

        token_embeddings = self._get_long_token_embeddings(inputs)

        # task prefix was added for Jina v3, correct for that
        token_offsets = inputs["offset_mapping"].squeeze(0).numpy() - len(task_prefix)
        token_chunk_annotations = self._map_chunks_to_tokens(
            chunk_annotations, token_offsets
        )

        pooled_embeddings = [
            token_embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in token_chunk_annotations
        ]
        pooled_embeddings = [
            F.normalize(embedding, p=2, dim=0).detach().cpu().tolist()
            for embedding in pooled_embeddings
        ]

        return pooled_embeddings

    def calculate_embeddings(
        self, input_text: str, task="retrieval.passage"
    ) -> list[float]:
        """Use Embedding Model directly"""
        embeddings = self.embedding_model.encode([input_text], task=task)[0]
        return embeddings
