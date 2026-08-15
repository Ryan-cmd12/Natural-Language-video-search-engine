from pathlib import Path

import cv2
import numpy as np


class WindowFrameSampler:

    def __init__(
        self,
        sample_count: int = 8,
    ):

        self.sample_count = (
            sample_count
        )

    def sample(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_dir: str,
    ) -> list[str]:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(
            video_path
        )

        if not capture.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:

            capture.release()

            raise RuntimeError(
                "Invalid video FPS."
            )

        start_frame = max(
            0,
            int(
                start_time
                *
                fps
            ),
        )

        end_frame = min(
            total_frames - 1,
            int(
                end_time
                *
                fps
            ),
        )

        if (
            end_frame
            <
            start_frame
        ):

            capture.release()

            return []

        available_frames = (
            end_frame
            -
            start_frame
            +
            1
        )

        count = min(
            self.sample_count,
            available_frames,
        )

        frame_indices = (
            np.linspace(
                start_frame,
                end_frame,
                num=count,
                dtype=int,
            )
            .tolist()
        )

        #
        # np.linspace can theoretically
        # produce duplicate integer indices.
        #

        frame_indices = list(
            dict.fromkeys(
                frame_indices
            )
        )

        frame_paths = []

        for frame_index in (
            frame_indices
        ):

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_index,
            )

            success, frame = (
                capture.read()
            )

            if not success:
                continue

            frame_path = (
                output_dir
                /
                (
                    f"frame_"
                    f"{frame_index:06d}"
                    f".jpg"
                )
            )

            cv2.imwrite(
                str(frame_path),
                frame,
            )

            frame_paths.append(
                str(frame_path)
            )

        capture.release()

        return frame_paths