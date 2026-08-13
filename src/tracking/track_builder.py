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
        direction:str = "forward",
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
            # Seed the tracker
            # --------------------------------

            initial = (
                self.tracker
                .add_text_prompt(
                    session_id=
                        session_id,

                    frame_index=
                        prompt_frame,

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
                "Initial objects:",
                len(initial_object_ids)
            )

            if len(initial_object_ids) == 0:

                print(
                    "No objects found on "
                    f"prompt frame {prompt_frame}. "
                    "Skipping propagation."
                )

                return []

            print(
                "Initial objects:",
                len(
                    initial_outputs.get(
                        "out_obj_ids",
                        [],
                    )
                )
            )

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
                        prompt_frame,

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