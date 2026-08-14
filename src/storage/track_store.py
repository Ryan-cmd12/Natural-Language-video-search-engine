import json
import re

from pathlib import Path

from src.models.tracked_object import (
    ObjectTrack,
    TrackPoint,
)

from src.models.track_index import (
    TrackIndexInfo,
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

    # ==================================================
    # PATH / LABEL HELPERS
    # ==================================================

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

    # ==================================================
    # CACHE OPERATIONS
    # ==================================================

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

    # ==================================================
    # OBJECT CONVERSION
    # ==================================================

    def _dict_to_track(
        self,
        data: dict,
    ) -> ObjectTrack:

        points = []

        for point_data in data.get(
            "points",
            []
        ):

            point = TrackPoint(
                frame_index=
                    point_data["frame_index"],

                timestamp=
                    point_data["timestamp"],

                x=
                    point_data["x"],

                y=
                    point_data["y"],

                width=
                    point_data["width"],

                height=
                    point_data["height"],
            )

            points.append(
                point
            )

        return ObjectTrack(
            track_id=
                data["track_id"],

            video_id=
                data["video_id"],

            label=
                data["label"],

            start_frame=
                data["start_frame"],

            end_frame=
                data["end_frame"],

            start_time=
                data["start_time"],

            end_time=
                data["end_time"],

            points=
                points,
        )

    # ==================================================
    # LOAD TRACK OBJECTS
    # ==================================================

    def load_tracks(
        self,
        video_id: str,
        label: str,
    ) -> list[ObjectTrack]:

        if not self.exists(
            video_id=video_id,
            label=label,
        ):

            return []

        data = self.load(
            video_id=video_id,
            label=label,
        )

        track_data = data.get(
            "tracks",
            []
        )

        return [
            self._dict_to_track(
                item
            )
            for item in track_data
        ]

    # ==================================================
    # SEARCH
    # ==================================================

    def search_tracks(
        self,
        video_id: str,
        label: str,
    ) -> list[ObjectTrack]:

        requested_slug = (
            self._slugify(
                label
            )
        )

        # ----------------------------------------------
        # 1. Exact cache match
        # ----------------------------------------------

        exact_tracks = (
            self.load_tracks(
                video_id=video_id,
                label=label,
            )
        )

        if exact_tracks:

            print(
                f"Exact track cache match: "
                f"{label}"
            )

            return exact_tracks

        # ----------------------------------------------
        # 2. Search available indexes
        # ----------------------------------------------

        matches = []

        pattern = (
            f"{video_id}__*.json"
        )

        for path in (
            self.root_dir.glob(
                pattern
            )
        ):

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

            except (
                json.JSONDecodeError,
                OSError,
            ):

                continue

            stored_label = (
                data.get(
                    "label",
                    ""
                )
            )

            if not stored_label:
                continue

            stored_slug = (
                self._slugify(
                    stored_label
                )
            )

            # ------------------------------------------
            # Basic V0.4 matching
            #
            # car     -> red car
            # red car -> red car
            #
            # NOTE:
            # this will be replaced by entity /
            # attribute resolution later.
            # ------------------------------------------

            if not self._labels_match(
                requested_slug,
                stored_slug,
            ):
                continue

            track_data = (
                data.get(
                    "tracks",
                    []
                )
            )

            tracks = [
                self._dict_to_track(
                    item
                )
                for item
                in track_data
            ]

            matches.extend(
                tracks
            )

        return matches

    # ==================================================
    # LABEL MATCHING
    # ==================================================

    def _labels_match(
        self,
        requested: str,
        stored: str,
    ) -> bool:

        if requested == stored:
            return True

        requested_tokens = set(
            requested.split("_")
        )

        stored_tokens = set(
            stored.split("_")
        )

        # Example:
        #
        # requested = car
        # stored    = red_car
        #
        # {"car"} <= {"red", "car"}
        #
        # True

        if requested_tokens.issubset(
            stored_tokens
        ):
            return True

        return False


    def list_indexes(
        self,
        video_id: str,
    ) -> list[TrackIndexInfo]:

        indexes = []

        pattern = (
            f"{video_id}__*.json"
        )

        for path in self.root_dir.glob(
            pattern
        ):

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

            except (
                OSError,
                json.JSONDecodeError,
            ):

                continue

            if (
                data.get("video_id")
                != video_id
            ):
                continue

            indexes.append(
                TrackIndexInfo(
                    video_id=
                        data["video_id"],

                    label=
                        data.get(
                            "label",
                            "",
                        ),

                    path=
                        str(path),

                    track_count=
                        data.get(
                            "track_count",
                            len(
                                data.get(
                                    "tracks",
                                    [],
                                )
                            ),
                        ),

                    fps=
                        data.get(
                            "fps"
                        ),

                    build_info=
                        data.get(
                            "build_info",
                            {},
                        ),
                )
            )

        return indexes

    def get_video_path(
    self,
    video_id: str,
    ) -> str | None:

        pattern = (
            f"{video_id}__*.json"
        )

        for path in self.root_dir.glob(
            pattern
        ):

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

            except (
                OSError,
                json.JSONDecodeError,
            ):

                continue

            if (
                data.get("video_id")
                != video_id
            ):
                continue

            video_path = (
                data.get(
                    "video_path"
                )
            )

            if video_path:
                return video_path

        return None