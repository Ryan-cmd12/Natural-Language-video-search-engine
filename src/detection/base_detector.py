from abc import ABC, abstractmethod

from src.models.detection import (
    ObjectDetection,
)


class ObjectDetector(ABC):

    @abstractmethod
    def detect(
        self,
        image_path: str,
        labels: list[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[ObjectDetection]:

        raise NotImplementedError