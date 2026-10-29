from __future__ import annotations

from collections.abc import Sequence


class SentenceTransformerEmbedder:
    """Local embedding provider with lazy loading of the model dependency."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "RAG dependencies are missing. Install with: "
                "python -m pip install -e \".[rag]\""
            ) from error
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(
                list(texts), convert_to_numpy=True, normalize_embeddings=True
            )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        if hasattr(self.model, "encode_query"):
            embedding = self.model.encode_query(
                text, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            embedding = self.model.encode(
                [text], convert_to_numpy=True, normalize_embeddings=True
            )[0]
        return embedding.tolist()