'''
split the sample frames into 3 batches, the first, middle and last frames
'''

from dataclasses import dataclass, field

import numpy as np

from src.detection.sam3_detector import (
    Sam3Detector,
)


@dataclass
class FrameGrounding:
    segment_frame_index: int
    frame_id: int | None
    frame_path: str

    detections: list = field(
        default_factory=list
    )


@dataclass
class SegmentGroundingResult:
    matched: bool

    prompt: str

    best_score: float

    frames_checked: int
    frames_with_detections: int

    grounded_frames: list[
        FrameGrounding
    ]


class Sam3SegmentGrounder:

    def __init__(
        self,
        max_frames,
        detector: Sam3Detector,
        min_score: float = 0.30,
    ):
        self.detector = detector
        self.max_frames = max_frames
        self.min_score = min_score

    def _select_indices(
        self,
        frame_count: int,
    ) -> list[int]:

        if frame_count <= 0:
            return []

        if frame_count <= self.max_frames:

            return list(
                range(frame_count)
            )

        indices = np.linspace(
            0,
            frame_count - 1,
            num=self.max_frames,
            dtype=int,
        )

        return list(
            dict.fromkeys(
                indices.tolist()
            )
        )

    def ground(
        self,
        segment: dict,
        prompt: str,
    ) -> SegmentGroundingResult:

        frame_paths = segment[
            "frame_paths"
        ]

        frame_ids = segment.get(
            "frame_ids",
            [],
        )

        selected_indices = (
            self._select_indices(
                len(frame_paths)
            )
        )
        print("\n[SAM GROUNDING]")
        print("prompt:", prompt)
        print("total frame_paths:", len(frame_paths))
        print("max_frames:", self.max_frames)
        print("selected_indices:", selected_indices)
        print("frames checked:", len(selected_indices))

        grounded_frames = []

        best_score = 0.0
        frames_with_detections = 0

        for index in selected_indices:

            frame_path = frame_paths[
                index
            ]
            print(
                f"[SAM] checking index={index} "
                f"path={frame_path}"
            )


            frame_id = (
                frame_ids[index]
                if index < len(frame_ids)
                else None
            )

            detections = (
                self.detector.detect(
                    image_path=frame_path,
                    prompt=prompt,
                    min_score=
                        self.min_score,
                )
            )

            if detections:
                print(
                f"[SAM] detections={len(detections)}"
            )


                frames_with_detections += 1

                best_score = max(
                    best_score,
                    max(
                        detection.score
                        for detection
                        in detections
                    ),
                )

            grounded_frames.append(
                FrameGrounding(
                    segment_frame_index=
                        index,

                    frame_id=
                        frame_id,

                    frame_path=
                        frame_path,

                    detections=
                        detections,
                )
            )

        return SegmentGroundingResult(
            matched=(
                frames_with_detections > 0
            ),

            prompt=prompt,

            best_score=
                best_score,

            frames_checked=
                len(selected_indices),

            frames_with_detections=
                frames_with_detections,

            grounded_frames=
                grounded_frames,
        )