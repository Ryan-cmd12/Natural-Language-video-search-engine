from collections import defaultdict

from src.models.segment import VideoSegment


def create_fixed_segments(
    frames: list[dict],
    video_id: str,
    video_duration: float,
    segment_seconds: float = 5.0,
) -> list[VideoSegment]:

    if segment_seconds <= 0:
        raise ValueError(
            "segment_seconds must be greater than 0"
        )

    grouped_frames = defaultdict(list)

    # --------------------------------
    # Assign each sampled frame
    # to a time window.
    #
    # Example:
    #
    # 0-5 seconds  -> segment 0
    # 5-10 seconds -> segment 1
    # ...
    # --------------------------------

    for frame in frames:

        timestamp = float(
            frame["timestamp"]
        )

        segment_id = int(
            timestamp // segment_seconds
        )

        grouped_frames[
            segment_id
        ].append(frame)

    segments = []

    for segment_id in sorted(
        grouped_frames.keys()
    ):

        segment_frames = grouped_frames[
            segment_id
        ]

        if not segment_frames:
            continue

        start_time = (
            segment_id
            * segment_seconds
        )

        end_time = min(
            start_time + segment_seconds,
            video_duration,
        )

        # Middle frame gives us a cheap
        # representative image.
        middle_index = (
            len(segment_frames) // 2
        )

        representative_frame = (
            segment_frames[middle_index]
        )

        segment = VideoSegment(
            segment_id=segment_id,
            video_id=video_id,

            start_time=start_time,
            end_time=end_time,

            frame_ids=[
                frame["frame_id"]
                for frame in segment_frames
            ],

            frame_paths=[
                frame["image_path"]
                for frame in segment_frames
            ],

            representative_frame_path=
                representative_frame[
                    "image_path"
                ],
        )

        segments.append(segment)

    return segments