from collections.abc import Callable

import numpy as np
from sentence_transformers import SentenceTransformer

from doc_chat.models import Chunk


class Embedder:

    def __init__(self, backbone_name: str) -> None:
        if backbone_name != "BAAI/bge-small-en-v1.5":
            raise ValueError(
                "backbone_name must be 'BAAI/bge-small-en-v1.5'"
            )
        
        self.backbone_name = backbone_name
        self.backbone = SentenceTransformer(backbone_name)


    def embed(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return np.empty((0, self.backbone.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = self.backbone.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)