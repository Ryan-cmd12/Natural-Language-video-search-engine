from dataclasses import dataclass, asdict


@dataclass
class VideoFrame:
    frame_id: int
    video_id: str
    frame_number: int
    timestamp: float
    image_path: str

    def to_dict(self) -> dict:
        return asdict(self)