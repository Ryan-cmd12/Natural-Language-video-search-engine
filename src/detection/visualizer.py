from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
)

from src.models.detection import (
    ObjectDetection,
)


def draw_detections(
    image_path: str,
    detections: list[ObjectDetection],
    output_path: str,
):

    image = (
        Image
        .open(image_path)
        .convert("RGB")
    )

    draw = ImageDraw.Draw(
        image
    )

    for detection in detections:

        box = [
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        ]

        draw.rectangle(
            box,
            outline="red",
            width=3,
        )

        label = (
            f"{detection.label} "
            f"{detection.score:.2f}"
        )

        text_position = (
            detection.x1,
            max(
                0,
                detection.y1 - 15,
            ),
        )

        draw.text(
            text_position,
            label,
            fill="red",
        )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output
    )

    image.close()