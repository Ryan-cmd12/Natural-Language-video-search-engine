import numpy as np

from sentence_transformers import SentenceTransformer


class TextEmbedder:

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        device: str = "cpu",
    ):
        print(
            f"Loading text embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.astype(
            "float32"
        )

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:

        return self.encode(
            [query]
        )