from dataclasses import dataclass, asdict


@dataclass
class VideoSegment:
    segment_id: int
    video_id: str

    start_time: float
    end_time: float

    frame_ids: list[int]
    frame_paths: list[str]

    representative_frame_path: str

    caption: str = ""

    def to_dict(self) -> dict:
        return asdict(self)



'''
Example of how a segment would look like
VideoSegment(
    segment_id=12,
    video_id="test",

    start_time=60.0,
    end_time=65.0,

    frame_ids=[
        60,
        61,
        62,
        63,
        64,
    ],

    frame_paths=[
        "frame_000060.jpg",
        "frame_000061.jpg",
        "frame_000062.jpg",
        "frame_000063.jpg",
        "frame_000064.jpg",
    ],

    representative_frame_path=
        "frame_000062.jpg",
)
'''