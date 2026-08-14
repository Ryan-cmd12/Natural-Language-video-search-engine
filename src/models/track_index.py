from dataclasses import dataclass, field


@dataclass
class TrackIndexInfo:

    video_id: str
    label: str
    path: str

    track_count: int

    fps: float | None = None

    build_info: dict = field(
        default_factory=dict
    )