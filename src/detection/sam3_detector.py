from pathlib import Path

from PIL import Image

import torch 
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from src.models.detection import ObjectDetection

class Sam3Detector:
    def __init__(self):
        print("Loading SAM 3 image model...")

        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

    def detect(
        self,
        image_path: str,
        prompt: str,
        min_score: float = 0.3,
    ) -> list[ObjectDetection]:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(path).convert("RGB")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)

            output = self.processor.set_text_prompt(
                state=inference_state,
                prompt=prompt,
            )

        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])

        detections = []

        for mask, box, score in zip(masks, boxes, scores):
            score_value = float(score)

            if score_value < min_score:
                continue

            x1, y1, x2, y2 = [float(v) for v in box]

            detections.append(
                ObjectDetection(
                    label=prompt,
                    score=score_value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_path=str(path),
                    mask=mask,
                )
            )

        image.close()
        return detections