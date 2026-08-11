from pathlib import Path

import numpy as np
import torch

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel


class CLIPEmbedder:

    def __init__(
        self,
        model_name: str = (
            "openai/clip-vit-base-patch32"
        ),
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        print(
            f"Loading CLIP model: {model_name}"
        )

        self.model = (
            CLIPModel
            .from_pretrained(model_name)
            .to(self.device)
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(model_name)
        )

        self.model.eval()

    @staticmethod
    def _extract_features(output):
        """
        Handles both older and newer
        Transformers return formats.
        """

        if torch.is_tensor(output):
            return output

        if hasattr(output, "pooler_output"):
            return output.pooler_output

        raise TypeError(
            "Unknown CLIP feature output type: "
            f"{type(output)}"
        )

    @staticmethod
    def _normalize(
        embeddings: np.ndarray,
    ) -> np.ndarray:

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        return embeddings / norms

    def encode_images(
        self,
        image_paths: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:

        all_embeddings = []

        for start in tqdm(
            range(0, len(image_paths), batch_size),
            desc="Embedding frames",
        ):
            batch_paths = image_paths[
                start:start + batch_size
            ]

            images = []

            for image_path in batch_paths:

                path = Path(image_path)

                image = (
                    Image
                    .open(path)
                    .convert("RGB")
                )

                images.append(image)

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )

            pixel_values = (
                inputs["pixel_values"]
                .to(self.device)
            )

            with torch.inference_mode():

                output = (
                    self.model
                    .get_image_features(
                        pixel_values=pixel_values
                    )
                )

                features = (
                    self._extract_features(
                        output
                    )
                )

            features = (
                features
                .detach()
                .cpu()
                .numpy()
                .astype("float32")
            )

            all_embeddings.append(
                features
            )

            for image in images:
                image.close()

        embeddings = np.vstack(
            all_embeddings
        )

        embeddings = self._normalize(
            embeddings
        )

        return embeddings.astype(
            "float32"
        )

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )

        input_ids = (
            inputs["input_ids"]
            .to(self.device)
        )

        attention_mask = (
            inputs["attention_mask"]
            .to(self.device)
        )

        with torch.inference_mode():

            output = (
                self.model
                .get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            )

            features = (
                self._extract_features(
                    output
                )
            )

        embeddings = (
            features
            .detach()
            .cpu()
            .numpy()
            .astype("float32")
        )

        embeddings = self._normalize(
            embeddings
        )

        return embeddings.astype(
            "float32"
        )