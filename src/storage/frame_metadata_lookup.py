import json

from pathlib import Path


class FrameMetadataLookup:

    def __init__(
        self,
        video_id: str,
        metadata_dir: str = "data/metadata",
    ):

        metadata_path = (
            Path(metadata_dir)
            /
            f"{video_id}.json"
        )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"V0.1 metadata not found: "
                f"{metadata_path}"
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            self.metadata = json.load(
                f
            )

        self.video = (
            self.metadata["video"]
        )

        self.frames = (
            self.metadata["frames"]
        )

        self.by_frame_id = {
            int(frame["frame_id"]):
                frame

            for frame
            in self.frames
        }

    @property
    def video_path(
        self,
    ) -> str:

        return self.video[
            "video_path"
        ]

    @property
    def fps(
        self,
    ) -> float:

        return float(
            self.video["fps"]
        )

    def get(
        self,
        frame_id: int,
    ) -> dict:

        if frame_id not in (
            self.by_frame_id
        ):

            raise KeyError(
                f"Unknown sampled "
                f"frame_id: {frame_id}"
            )

        return self.by_frame_id[
            frame_id
        ]

    def get_video_frame_number(
        self,
        frame_id: int,
    ) -> int:

        frame = self.get(
            frame_id
        )

        return int(
            frame["frame_number"]
        )