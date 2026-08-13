import json
import re

from pathlib import Path

from src.models.tracked_object import (
    ObjectTrack,
)


class TrackStore:

    def __init__(
        self,
        root_dir: str = "data/tracks",
    ):
        self.root_dir = Path(
            root_dir
        )

        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _slugify(
        text: str,
    ) -> str:

        text = (
            text
            .lower()
            .strip()
        )

        text = re.sub(
            r"[^a-z0-9]+",
            "_",
            text,
        )

        return text.strip("_")

    def get_path(
        self,
        video_id: str,
        label: str,
    ) -> Path:

        slug = self._slugify(
            label
        )

        return (
            self.root_dir
            /
            f"{video_id}__{slug}.json"
        )

    def exists(
        self,
        video_id: str,
        label: str,
    ) -> bool:

        return self.get_path(
            video_id=video_id,
            label=label,
        ).exists()

    def save(
        self,
        video_id: str,
        label: str,
        video_path: str,
        fps: float,
        tracks: list[ObjectTrack],
        build_info: dict | None = None,
    ) -> Path:

        output_path = self.get_path(
            video_id=video_id,
            label=label,
        )

        data = {
            "schema_version": 1,

            "video_id":
                video_id,

            "video_path":
                video_path,

            "label":
                label,

            "fps":
                fps,

            "track_count":
                len(tracks),

            "build_info":
                build_info or {},

            "tracks": [
                track.to_dict()
                for track in tracks
            ],
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
            )

        return output_path

    def load(
        self,
        video_id: str,
        label: str,
    ) -> dict:

        path = self.get_path(
            video_id=video_id,
            label=label,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Track cache not found: "
                f"{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    def delete(
        self,
        video_id: str,
        label: str,
    ) -> bool:

        path = self.get_path(
            video_id=video_id,
            label=label,
        )

        if not path.exists():
            return False

        path.unlink()

        return True