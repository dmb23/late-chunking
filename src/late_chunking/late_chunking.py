import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sentence_transformers import SentenceTransformer


class LateEmbedder:
    """Calculate chunk embeddings via late chunking.

    Late Chunking is an embedding method that keeps long context information
    inside chunk embeddings. Token-level embeddings are calculated for the
    full text initially to keep context information in a long-context
    embedding model. Only after that the (tokenized & embedded) text is
    cut into chunks and the embeddings are averaged.

    Args:
        model_name: name of a Sentence Transformer model
        model_kwargs:  dict of kwargs to be passed to model instantiation
        max_seq_length: maximal sequence length the model can process
        max_seq_overlap: overlap for chunking texts longer than the maximal sequence length
        document_prefix: Prefix that should be added before all documents that are encoded for retrieval
    """

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 8192,
        max_seq_overlap: int = 500,
        document_prefix: str = "",
        model_kwargs: dict | None = None,
    ) -> None:
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.embedder = SentenceTransformer(model_name, **model_kwargs)
        self.max_seq_length = max_seq_length
        self.max_seq_overlap = max_seq_overlap
        self._document_prefix = document_prefix

    def calculate_late_embeddings(
        self, input_text: str, chunk_annotations: list[tuple[int, int]]
    ) -> np.typing.NDArray:
        """Calculate late embeddings for chunks in a longer text

        Args:
            input_text: a full text for which embeddings of chunks should be calculated
            chunk_annotations: a list of (chunk_start_index, chunk_end_index) that identify the chunks for the text

        Returns:
            np.ndarray of shape [number of chunks, embedding dimension]
        """
        token_embeddings, token_offsets = self._get_long_token_embeddings(input_text)

        token_chunk_annotations = self._map_chunks_to_tokens(
            chunk_annotations, token_offsets
        )

        pooled_embeddings = [
            token_embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in token_chunk_annotations
        ]
        pooled_embeddings = [
            F.normalize(embedding, p=2, dim=0).detach().cpu().numpy()[None, :]
            for embedding in pooled_embeddings
        ]
        pooled_embeddings = np.concat(pooled_embeddings, 0)

        return pooled_embeddings

    def _get_token_lengths(self) -> tuple[int, int, int]:
        """Calculate the number of tokens for prefix and special tokens.

        Returns:
            tuple containing:
            - prefix_length: number of tokens in the prefix
            - leading_special_tokens: number of special tokens added at the start
            - trailing_special_tokens: number of special tokens added at the end
        """
        # Get prefix length without special tokens
        if self._document_prefix:
            prefix_tokens = self.embedder.tokenizer(
                self._document_prefix, add_special_tokens=False
            )
            prefix_length = len(prefix_tokens["input_ids"])
        else:
            prefix_length = 0

        text = "This is an example text."
        # Get number of special tokens by comparing with/without
        text_tokens_with_special = self.embedder.tokenizer(text)
        text_tokens_without_special = self.embedder.tokenizer(
            text, add_special_tokens=False
        )

        # Compare token IDs to identify special tokens
        tokens_with = text_tokens_with_special["input_ids"]
        tokens_without = text_tokens_without_special["input_ids"]

        # Count leading special tokens by comparing from the start
        leading_special_tokens = 0
        while tokens_with[leading_special_tokens] != tokens_without[0]:
            leading_special_tokens += 1

        # Count trailing special tokens by comparing from the end
        trailing_special_tokens = 0
        while tokens_with[-(trailing_special_tokens + 1)] != tokens_without[-1]:
            trailing_special_tokens += 1

        return prefix_length, leading_special_tokens, trailing_special_tokens

    def _get_long_token_embeddings(
        self, full_text: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Handle inputs that might be longer than the embedding model can handle.

        Args:
            full_text: Text to tokenize. Can be longer than the maximum sequence length of the embedding model.

        Returns:
            token_embeddings: embedding vectors for each token in the text
            token_offsets: mapping of the tokens to their positions in the original text
        """
        prefix_length, leading_special, trailing_special = self._get_token_lengths()

        # Tokenize full text to get offsets
        base_tokens = self.embedder.tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            verbose=False,
            add_special_tokens=False,
        )
        n_tokens = base_tokens["input_ids"].numel()
        token_offsets = base_tokens["offset_mapping"].squeeze()

        device = self.embedder.device

        if (
            n_tokens
            <= self.max_seq_length - prefix_length - leading_special - trailing_special
        ):
            tokens = self.embedder.tokenizer(
                self._document_prefix + full_text,
                return_tensors="pt",
            )

            for key in tokens:
                tokens[key] = (
                    tokens[key].to(device)
                    if isinstance(tokens[key], torch.Tensor)
                    else tokens[key]
                )
            with torch.no_grad():
                model_output = self.embedder(tokens)

            # Exclude prefix tokens and leading special tokens from output
            start_idx = leading_special + prefix_length
            end_idx = -trailing_special if trailing_special > 0 else None
            token_embeddings = model_output["token_embeddings"].squeeze(0)[
                start_idx:end_idx
            ]
        else:
            chunk_embeddings = []

            # Create chunks with overlap, adding prefix to each
            for chunk_start in range(
                0,
                n_tokens,
                self.max_seq_length
                - self.max_seq_overlap
                - prefix_length
                - leading_special
                - trailing_special,
            ):
                chunk_end = min(
                    chunk_start
                    + self.max_seq_length
                    - prefix_length
                    - leading_special
                    - trailing_special,
                    n_tokens,
                )
                chunk_text = (
                    self._document_prefix
                    + full_text[
                        token_offsets[chunk_start, 0] : token_offsets[chunk_end - 1, 1]
                    ]
                )
                chunk = self.embedder.tokenizer(chunk_text, return_tensors="pt")
                chunk = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in chunk.items()
                }
                with torch.no_grad():
                    chunk_output = self.embedder(chunk)
                # Exclude prefix tokens and special tokens
                start_idx = leading_special + prefix_length
                end_idx = -trailing_special if trailing_special > 0 else None
                chunk_embeddings.append(
                    chunk_output["token_embeddings"].squeeze(0)[start_idx:end_idx]
                )

            token_embeddings = torch.zeros(
                (n_tokens, chunk_embeddings[0].shape[1]),
                dtype=chunk_embeddings[0].dtype,
            ).to(device)

            counts = torch.zeros(n_tokens, dtype=torch.int).to(device)

            pos = 0
            for chunk_emb in chunk_embeddings:
                chunk_size = chunk_emb.shape[0]
                token_embeddings[pos : pos + chunk_size] += chunk_emb
                counts[pos : pos + chunk_size] += 1
                pos += (
                    self.max_seq_length
                    - self.max_seq_overlap
                    - prefix_length
                    - leading_special
                    - trailing_special
                )

            # Average overlapping regions
            token_embeddings = token_embeddings / counts.unsqueeze(1)

        return token_embeddings, token_offsets

    def _map_chunks_to_tokens(
        self, chunk_annotations: list[tuple[int, int]], token_offsets: torch.Tensor
    ) -> list[tuple[int, int]]:
        """Translate chunk boundaries from text to token space.

        Args:
            chunk_annotations: list of (chunk_start_index, chunk_end_index), positions measured in the text
            token_offsets: tensor of (token_start_index, token_end_index), positions measured in the text

        Returns:
            list of (chunk_start_token_index, chunk_end_token_index) for the tokenized text
        """
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

            token_chunk_annotations.append((start_token.item(), end_token.item()))

        return token_chunk_annotations
