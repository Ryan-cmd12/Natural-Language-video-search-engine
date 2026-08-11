from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    video_id: str
    video_path: str

    fps: float
    frame_count: int
    duration: float

    width: int
    height: int


def get_video_metadata(video_path: str) -> VideoMetadata:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Video does not exist: {video_path}"
        )

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV could not open video: {video_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS reported for video: {fps}"
        )

    duration = frame_count / fps

    return VideoMetadata(
        video_id=path.stem,
        video_path=str(path),
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        width=width,
        height=height,
    )