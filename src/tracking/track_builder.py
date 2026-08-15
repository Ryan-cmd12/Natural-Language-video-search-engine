from collections import defaultdict
from pathlib import Path

import numpy as np

from src.models.tracked_object import (
    ObjectTrack,
    TrackPoint,
)

from src.tracking.sam3_video_tracker import (
    Sam3VideoTracker,
)


class TrackBuilder:

    def __init__(
        self,
        tracker: Sam3VideoTracker,
    ):
        self.tracker = tracker

    def build_tracks(
        self,
        video_path: str,
        video_id: str,
        prompt: str,
        fps: float,
        prompt_frame: int = 0,
        output_prob_thresh: float = 0.5,
        max_frames: int | None = None,
        total_frames: int | None = None,
        scan_interval: int | None = None,
        direction: str = "both",
    ) -> list[ObjectTrack]:

        if fps <= 0:
            raise ValueError(
                "FPS must be greater than 0."
            )

        session_id = (
            self.tracker.start_session(
                resource_path=
                    video_path
            )
        )

        print(
            f"Started SAM 3 session: "
            f"{session_id}"
        )

        try:
            # --------------------------------
            # Find a seed frame
            # --------------------------------

            if scan_interval is not None:

                if scan_interval <= 0:
                    raise ValueError(
                        "scan_interval must be greater than 0."
                    )

                if total_frames is None:
                    raise ValueError(
                        "total_frames is required "
                        "when scan_interval is used."
                    )

                frames_to_try = list(
                    range(
                        0,
                        total_frames,
                        scan_interval,
                    )
                )

                # Make sure we also test the final frame.
                last_frame = total_frames - 1

                if (
                    frames_to_try
                    and frames_to_try[-1]
                    != last_frame
                ):
                    frames_to_try.append(
                        last_frame
                    )

            else:

                frames_to_try = [
                    prompt_frame
                ]


            seed_frame = None

            for frame_index in frames_to_try:

                print(
                    f"Searching frame "
                    f"{frame_index} "
                    f"for '{prompt}'..."
                )

                initial = (
                    self.tracker
                    .add_text_prompt(
                        session_id=
                            session_id,

                        frame_index=
                            frame_index,

                        text=
                            prompt,

                        output_prob_thresh=
                            output_prob_thresh,
                    )
                )

                initial_outputs = (
                    initial.get(
                        "outputs",
                        {},
                    )
                )

                initial_object_ids = (
                    initial_outputs.get(
                        "out_obj_ids",
                        [],
                    )
                )

                if initial_object_ids is None:
                    initial_object_ids = []

                print(
                    f"Objects found: "
                    f"{len(initial_object_ids)}"
                )

                if len(initial_object_ids) > 0:

                    seed_frame = (
                        frame_index
                    )

                    print(
                        f"\nFound '{prompt}' "
                        f"on frame "
                        f"{seed_frame}."
                    )

                    break


            if seed_frame is None:

                print(
                    f"\nNo '{prompt}' objects "
                    f"found in scanned frames."
                )

                return []
            # --------------------------------
            # object_id -> TrackPoint[]
            # --------------------------------

            track_points = defaultdict(
                list
            )

            seen_frames = defaultdict(
                set
            )

            print(
                "\nPropagating tracks..."
            )

            for response in (
                self.tracker.propagate(
                    session_id=
                        session_id,

                    direction=
                        direction,

                    start_frame_index=
                        seed_frame,

                    max_frames=
                        max_frames,

                    output_prob_thresh=
                        output_prob_thresh,
                )
            ):

                frame_index = int(
                    response[
                        "frame_index"
                    ]
                )

                outputs = (
                    response.get(
                        "outputs",
                        {},
                    )
                )

                object_ids = (
                    outputs.get(
                        "out_obj_ids",
                        [],
                    )
                )

                boxes = (
                    outputs.get(
                        "out_boxes_xywh",
                        [],
                    )
                )

                if object_ids is None:
                    continue

                if boxes is None:
                    continue

                object_ids = np.asarray(
                    object_ids
                )

                boxes = np.asarray(
                    boxes
                )

                if len(object_ids) == 0:
                    continue

                timestamp = (
                    frame_index
                    / fps
                )

                for (
                    object_id,
                    box,
                ) in zip(
                    object_ids,
                    boxes,
                ):

                    object_id = int(
                        object_id
                    )

                    # Prevent accidental
                    # duplicate frame entries.
                    if (
                        frame_index
                        in seen_frames[
                            object_id
                        ]
                    ):
                        continue

                    if len(box) != 4:
                        continue

                    x, y, w, h = [
                        float(value)
                        for value
                        in box
                    ]

                    point = TrackPoint(
                        frame_index=
                            frame_index,

                        timestamp=
                            timestamp,

                        x=x,
                        y=y,
                        width=w,
                        height=h,
                    )

                    track_points[
                        object_id
                    ].append(
                        point
                    )

                    seen_frames[
                        object_id
                    ].add(
                        frame_index
                    )

            # --------------------------------
            # Turn raw points into tracks
            # --------------------------------

            tracks = []

            for (
                object_id,
                points,
            ) in track_points.items():

                if not points:
                    continue

                points.sort(
                    key=lambda point:
                        point.frame_index
                )

                first = points[0]
                last = points[-1]

                track = ObjectTrack(
                    track_id=
                        object_id,

                    video_id=
                        video_id,

                    label=
                        prompt,

                    start_frame=
                        first.frame_index,

                    end_frame=
                        last.frame_index,

                    start_time=
                        first.timestamp,

                    end_time=
                        last.timestamp,

                    points=
                        points,
                )

                tracks.append(
                    track
                )

            tracks.sort(
                key=lambda track:
                    track.start_frame
            )

            return tracks

        finally:

            self.tracker.close_session(
                session_id
            )