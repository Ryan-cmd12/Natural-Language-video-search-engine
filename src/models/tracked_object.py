from dataclasses import dataclass, asdict


@dataclass
class TrackPoint:
    frame_index: int
    timestamp: float

    # Keep SAM's native xywh format for now.
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def bbox_xywh(self) -> list[float]:
        return [
            self.x,
            self.y,
            self.width,
            self.height,
        ]


@dataclass
class ObjectTrack:
    track_id: int
    video_id: str
    label: str

    start_frame: int
    end_frame: int

    start_time: float
    end_time: float

    points: list[TrackPoint]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "video_id": self.video_id,
            "label": self.label,

            "start_frame": self.start_frame,
            "end_frame": self.end_frame,

            "start_time": self.start_time,
            "end_time": self.end_time,

            "points": [
                point.to_dict()
                for point in self.points
            ],
        }

    @property
    def duration(self) -> float:
        return (
            self.end_time
            - self.start_time
        )