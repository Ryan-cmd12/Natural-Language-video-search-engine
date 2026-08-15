from dataclasses import dataclass

import math

from src.models.tracked_object import (
    ObjectTrack,
    TrackPoint,
)


@dataclass
class SpatialRelationshipResult:

    status: str

    # MATCH
    # REJECT
    # UNVERIFIED

    confidence: float

    relationship: str

    subject_track_id: int | str | None
    object_track_id: int | str | None

    evaluated_frames: list[int]
    matching_frames: list[int]

    reason: str


class SpatialRelationshipFilter:

    SUPPORTED_RELATIONSHIPS = {
        "left_of",
        "right_of",
        "above",
        "below",
        "near",
        "overlapping",
    }

    def __init__(
        self,
        min_match_ratio: float = 0.6,
        min_common_frames: int = 2,
        directional_margin: float = 0.05,
        near_threshold: float = 1.5,
        overlap_iou_threshold: float = 0.05,
    ):
        self.min_match_ratio = min_match_ratio
        self.min_common_frames = min_common_frames
        self.directional_margin = directional_margin
        self.near_threshold = near_threshold
        self.overlap_iou_threshold = (
            overlap_iou_threshold
        )

    def evaluate(
        self,
        subject_track: ObjectTrack,
        object_track: ObjectTrack,
        relationship: str,
    ) -> SpatialRelationshipResult:

        relationship = (
            relationship
            .strip()
            .lower()
        )

        if (
            relationship
            not in
            self.SUPPORTED_RELATIONSHIPS
        ):
            raise ValueError(
                f"Unsupported spatial "
                f"relationship: {relationship}"
            )

        subject_points = (
            self._points_by_frame(
                subject_track
            )
        )

        object_points = (
            self._points_by_frame(
                object_track
            )
        )

        common_frames = sorted(
            set(subject_points)
            &
            set(object_points)
        )

        subject_track_id = (
            self._track_id(
                subject_track
            )
        )

        object_track_id = (
            self._track_id(
                object_track
            )
        )

        if (
            len(common_frames)
            <
            self.min_common_frames
        ):
            return SpatialRelationshipResult(
                status="UNVERIFIED",
                confidence=0.0,
                relationship=relationship,
                subject_track_id=(
                    subject_track_id
                ),
                object_track_id=(
                    object_track_id
                ),
                evaluated_frames=(
                    common_frames
                ),
                matching_frames=[],
                reason=(
                    "Not enough frames where "
                    "both tracks are visible."
                ),
            )

        matching_frames = []

        for frame_number in common_frames:

            subject_point = (
                subject_points[
                    frame_number
                ]
            )

            object_point = (
                object_points[
                    frame_number
                ]
            )

            if self._evaluate_frame(
                subject_point,
                object_point,
                relationship,
            ):
                matching_frames.append(
                    frame_number
                )

        match_ratio = (
            len(matching_frames)
            /
            len(common_frames)
        )

        if (
            match_ratio
            >=
            self.min_match_ratio
        ):
            status = "MATCH"

        else:
            status = "REJECT"

        return SpatialRelationshipResult(
            status=status,
            confidence=match_ratio,
            relationship=relationship,
            subject_track_id=(
                subject_track_id
            ),
            object_track_id=(
                object_track_id
            ),
            evaluated_frames=(
                common_frames
            ),
            matching_frames=(
                matching_frames
            ),
            reason=(
                f"Relationship "
                f"'{relationship}' matched "
                f"{len(matching_frames)} / "
                f"{len(common_frames)} "
                f"common frames."
            ),
        )

    def filter(
        self,
        subject_tracks: list[ObjectTrack],
        object_tracks: list[ObjectTrack],
        relationship: str,
    ) -> list[
        SpatialRelationshipResult
    ]:

        results = []

        for subject_track in subject_tracks:

            for object_track in object_tracks:

                if (
                    subject_track
                    is
                    object_track
                ):
                    continue

                result = self.evaluate(
                    subject_track,
                    object_track,
                    relationship,
                )

                if (
                    result.status
                    ==
                    "MATCH"
                ):
                    results.append(
                        result
                    )

        results.sort(
            key=lambda result: (
                result.confidence
            ),
            reverse=True,
        )

        return results

    def _points_by_frame(
        self,
        track: ObjectTrack,
    ) -> dict[int, TrackPoint]:

        result = {}

        for point in track.points:

            result[
                self._frame_number(point)
            ] = point

        return result

    def _frame_number(
        self,
        point: TrackPoint,
    ) -> int:

        for field in (
            "frame_number",
            "frame_index",
            "frame_idx",
        ):

            if hasattr(
                point,
                field,
            ):
                return int(
                    getattr(
                        point,
                        field,
                    )
                )

        raise AttributeError(
            "TrackPoint has no frame "
            "number field."
        )

    def _track_id(
        self,
        track: ObjectTrack,
    ) -> int:

        return track.track_id

    def _bbox(self,point: TrackPoint,) -> tuple[
        float,
        float,
        float,
        float,]:

        x1 = float(point.x)
        y1 = float(point.y)

        x2 = x1 + float(point.width)
        y2 = y1 + float(point.height)

        return (
            x1,
            y1,
            x2,
            y2,
        )

    def _center(
        self,
        point: TrackPoint,
    ):

        x1, y1, x2, y2 = (
            self._bbox(point)
        )

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        )

    def _size(
        self,
        point: TrackPoint,
    ):

        x1, y1, x2, y2 = (
            self._bbox(point)
        )

        return (
            max(0.0, x2 - x1),
            max(0.0, y2 - y1),
        )

    def _evaluate_frame(
        self,
        subject: TrackPoint,
        obj: TrackPoint,
        relationship: str,
    ):

        methods = {
            "left_of": self._is_left_of,
            "right_of": self._is_right_of,
            "above": self._is_above,
            "below": self._is_below,
            "near": self._is_near,
            "overlapping": (
                self._is_overlapping
            ),
        }

        return methods[
            relationship
        ](
            subject,
            obj,
        )

    def _is_left_of(
        self,
        subject,
        obj,
    ):

        sx, _ = self._center(subject)
        ox, _ = self._center(obj)

        sw, _ = self._size(subject)
        ow, _ = self._size(obj)

        reference = (
            sw + ow
        ) / 2

        margin = (
            reference
            *
            self.directional_margin
        )

        return sx < ox - margin

    def _is_right_of(
        self,
        subject,
        obj,
    ):

        sx, _ = self._center(subject)
        ox, _ = self._center(obj)

        sw, _ = self._size(subject)
        ow, _ = self._size(obj)

        reference = (
            sw + ow
        ) / 2

        margin = (
            reference
            *
            self.directional_margin
        )

        return sx > ox + margin

    def _is_above(
        self,
        subject,
        obj,
    ):

        _, sy = self._center(subject)
        _, oy = self._center(obj)

        _, sh = self._size(subject)
        _, oh = self._size(obj)

        reference = (
            sh + oh
        ) / 2

        margin = (
            reference
            *
            self.directional_margin
        )

        return sy < oy - margin

    def _is_below(
        self,
        subject,
        obj,
    ):

        _, sy = self._center(subject)
        _, oy = self._center(obj)

        _, sh = self._size(subject)
        _, oh = self._size(obj)

        reference = (
            sh + oh
        ) / 2

        margin = (
            reference
            *
            self.directional_margin
        )

        return sy > oy + margin

    def _is_near(
        self,
        subject,
        obj,
    ):

        sx, sy = self._center(subject)
        ox, oy = self._center(obj)

        distance = math.sqrt(
            (sx - ox) ** 2
            +
            (sy - oy) ** 2
        )

        sw, sh = self._size(subject)
        ow, oh = self._size(obj)

        subject_diag = math.sqrt(
            sw ** 2
            +
            sh ** 2
        )

        object_diag = math.sqrt(
            ow ** 2
            +
            oh ** 2
        )

        reference = (
            subject_diag
            +
            object_diag
        ) / 2

        if reference <= 0:
            return False

        normalized_distance = (
            distance
            /
            reference
        )

        return (
            normalized_distance
            <=
            self.near_threshold
        )

    def _is_overlapping(
        self,
        subject,
        obj,
    ):

        return (
            self._iou(
                subject,
                obj,
            )
            >=
            self.overlap_iou_threshold
        )

    def _iou(
        self,
        a,
        b,
    ):

        ax1, ay1, ax2, ay2 = (
            self._bbox(a)
        )

        bx1, by1, bx2, by2 = (
            self._bbox(b)
        )

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = (
            max(0.0, ax2 - ax1)
            *
            max(0.0, ay2 - ay1)
        )

        area_b = (
            max(0.0, bx2 - bx1)
            *
            max(0.0, by2 - by1)
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return (
            intersection
            /
            union
        )