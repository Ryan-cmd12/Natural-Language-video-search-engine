from pathlib import Path

import cv2
import numpy as np

from src.models.tracked_object import (
    ObjectTrack,
)


class TrackFrameSampler:

    def __init__(
        self,
        output_dir: str = "data/query_tmp",
        samples_per_track: int = 3,
        crop_padding: float = 0.20,
    ):

        self.output_dir = Path(
            output_dir
        )

        self.samples_per_track = (
            samples_per_track
        )

        self.crop_padding = (
            crop_padding
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==================================================
    # PUBLIC
    # ==================================================

    def build_contact_sheet(
        self,
        video_path: str,
        video_id: str,
        track: ObjectTrack,
    ) -> Path | None:

        if not track.points:
            return None

        selected_points = (
            self._sample_points(
                track
            )
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise RuntimeError(
                f"Could not open video: "
                f"{video_path}"
            )

        crops = []

        try:

            for point in selected_points:

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    point.frame_index,
                )

                success, frame = (
                    cap.read()
                )

                if not success:
                    continue

                crop = self._crop_track(
                    frame=frame,
                    point=point,
                )

                if crop is None:
                    continue

                crops.append(
                    crop
                )

        finally:

            cap.release()

        if not crops:
            return None

        sheet = self._make_contact_sheet(
            crops
        )

        track_dir = (
            self.output_dir
            /
            self._safe_name(video_id)
        )

        track_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            track_dir
            /
            (
                f"track_"
                f"{self._safe_name(str(track.track_id))}"
                f"_attributes.jpg"
            )
        )

        cv2.imwrite(
            str(output_path),
            sheet,
        )

        return output_path

    # ==================================================
    # SAMPLING
    # ==================================================

    def _sample_points(
        self,
        track: ObjectTrack,
    ):

        points = sorted(
            track.points,
            key=lambda p:
                p.frame_index,
        )

        count = min(
            self.samples_per_track,
            len(points),
        )

        if count == 1:
            return [
                points[len(points) // 2]
            ]

        indexes = np.linspace(
            0,
            len(points) - 1,
            count,
        )

        indexes = [
            int(round(index))
            for index in indexes
        ]

        return [
            points[index]
            for index in indexes
        ]

    # ==================================================
    # CROP
    # ==================================================

    def _crop_track(
        self,
        frame,
        point,
    ):

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x = float(point.x)
        y = float(point.y)

        width = float(
            point.width
        )

        height = float(
            point.height
        )

        #
        # Handle normalized coordinates too.
        #

        if (
            0 <= x <= 1.5
            and
            0 <= y <= 1.5
            and
            0 <= width <= 1.5
            and
            0 <= height <= 1.5
        ):

            x *= frame_width
            width *= frame_width

            y *= frame_height
            height *= frame_height

        padding_x = (
            width
            *
            self.crop_padding
        )

        padding_y = (
            height
            *
            self.crop_padding
        )

        x1 = int(
            max(
                0,
                x - padding_x,
            )
        )

        y1 = int(
            max(
                0,
                y - padding_y,
            )
        )

        x2 = int(
            min(
                frame_width,
                x + width + padding_x,
            )
        )

        y2 = int(
            min(
                frame_height,
                y + height + padding_y,
            )
        )

        if (
            x2 <= x1
            or
            y2 <= y1
        ):
            return None

        return frame[
            y1:y2,
            x1:x2
        ]

    # ==================================================
    # CONTACT SHEET
    # ==================================================

    def _make_contact_sheet(
        self,
        crops,
    ):

        target_height = 320

        resized = []

        for crop in crops:

            height, width = (
                crop.shape[:2]
            )

            scale = (
                target_height
                /
                max(height, 1)
            )

            new_width = int(
                width * scale
            )

            new_width = max(
                new_width,
                1,
            )

            resized_crop = (
                cv2.resize(
                    crop,
                    (
                        new_width,
                        target_height,
                    ),
                )
            )

            resized.append(
                resized_crop
            )

        return cv2.hconcat(
            resized
        )

    def _safe_name(
        self,
        text: str,
    ):

        return (
            str(text)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace(" ", "_")
        )