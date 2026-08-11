from pathlib import Path

import faiss
import numpy as np


class FaissIndex:

    def __init__(
        self,
        dimension: int,
    ):
        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

    def add(
        self,
        embeddings: np.ndarray,
    ):
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a "
                "2D numpy array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(
            embeddings
        )

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ):

        query_embedding = (
            np.ascontiguousarray(
                query_embedding,
                dtype=np.float32,
            )
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                k,
            )
        )

        return scores, indices

    def save(
        self,
        path: str,
    ):
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        path: str,
    ):

        index = faiss.read_index(
            str(path)
        )

        obj = cls.__new__(cls)

        obj.index = index
        obj.dimension = index.d

        return obj