from pathlib import Path

import torch

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)

from src.detection.base_detector import (
    ObjectDetector,
)

from src.models.detection import (
    ObjectDetection,
)


class GroundingDinoDetector(
    ObjectDetector
):

    def __init__(
        self,
        model_name: str = (
            "IDEA-Research/"
            "grounding-dino-base"
        ),
        device: str = "cpu",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(
            device
        )

        print(
            f"Loading Grounding DINO: "
            f"{model_name}"
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(
                model_name
            )
            .to(self.device)
        )

        self.model.eval()

    def detect(self,image_path: str,labels: list[str],box_threshold: float = 0.40,text_threshold: float = 0.30,) -> list[ObjectDetection]:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        clean_labels = [
            label.strip().lower()
            for label in labels
            if label.strip()
        ]

        if not clean_labels:
            raise ValueError(
                "At least one detection "
                "label is required."
            )

        image = (
            Image
            .open(path)
            .convert("RGB")
        )

        text_labels = [
            clean_labels
        ]

        inputs = self.processor(
            images=image,
            text=text_labels,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

        results = (
            self.processor
            .post_process_grounded_object_detection(
                outputs,

                input_ids=
                    inputs["input_ids"],

                threshold=
                    box_threshold,

                text_threshold=
                    text_threshold,

                target_sizes=[
                    (
                        image.height,
                        image.width,
                    )
                ],
            )
        )

        image.close()

        result = results[0]

        detections = []

        boxes = result[
            "boxes"
        ]

        scores = result[
            "scores"
        ]

        detected_labels = result.get(
            "text_labels",
            result.get(
                "labels",
                [],
            ),
        )

        for (
            box,
            score,
            label,
        ) in zip(
            boxes,
            scores,
            detected_labels,
        ):

            x1, y1, x2, y2 = [
                float(value)
                for value
                in box.tolist()
            ]

            detections.append(
                ObjectDetection(
                    label=str(label),

                    score=float(
                        score.item()
                    ),

                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,

                    image_path=
                        str(path),
                )
            )

        return detections