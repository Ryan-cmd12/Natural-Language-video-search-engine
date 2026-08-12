from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ObjectDetection:
    label: str
    score: float

    x1: float
    y1: float
    x2: float
    y2: float

    image_path: str

    frame_id: int | None = None
    timestamp: float | None = None

    #for sam3
    mask: Any | None = None

    def to_dict(self) -> dict:
        data = asdict(self)

        # Don't try to serialize the raw mask tensor directly
        if data["mask"] is not None:
            data["mask"] = "<mask>"
        return data

    @property
    def bbox(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

'''
Example of how an objectDetection shld look like:

ObjectDetection(
    label="red backpack",
    score=0.87,

    x1=412,
    y1=183,
    x2=601,
    y2=514,

    image_path="frame_00231.jpg",
)
'''