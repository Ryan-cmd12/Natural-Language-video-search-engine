from pathlib import Path

from PIL import Image

from src.models.detection import (
    ObjectDetection,
)


def crop_detection(
    detection: ObjectDetection,
    output_path: str,
    padding: float = 0.10,
) -> str:

    image = (
        Image
        .open(detection.image_path)
        .convert("RGB")
    )

    width, height = image.size

    box_width = (
        detection.x2
        - detection.x1
    )

    box_height = (
        detection.y2
        - detection.y1
    )

    pad_x = (
        box_width
        * padding
    )

    pad_y = (
        box_height
        * padding
    )

    x1 = max(
        0,
        detection.x1 - pad_x,
    )

    y1 = max(
        0,
        detection.y1 - pad_y,
    )

    x2 = min(
        width,
        detection.x2 + pad_x,
    )

    y2 = min(
        height,
        detection.y2 + pad_y,
    )

    crop = image.crop(
        (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
    )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        output
    )

    crop.close()
    image.close()

    return str(output)