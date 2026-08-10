import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self, backbone_name: str) -> None:
        if backbone_name != "BAAI/bge-small-en-v1.5":
            raise ValueError(
                "backbone_name must be 'BAAI/bge-small-en-v1.5'"
            )
        
        self.backbone_name = backbone_name
        self.backbone = SentenceTransformer(backbone_name)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
            if not texts:
                return np.empty((0, self.backbone.get_sentence_embedding_dimension()), dtype=np.float32)
    
            embeddings = self.backbone.encode(
                texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)