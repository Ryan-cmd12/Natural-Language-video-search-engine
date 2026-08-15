import argparse
import json

import cv2

from src.models.tracked_object import (
    ObjectTrack,
    TrackPoint,
)


def load_tracks(
    path: str,
) -> list[ObjectTrack]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):
        raw_tracks = data

    elif (
        isinstance(data, dict)
        and
        "tracks" in data
    ):
        raw_tracks = data["tracks"]

    else:
        raw_tracks = [data]

    tracks = []

    for raw in raw_tracks:

        points = [
            TrackPoint(
                frame_index=point[
                    "frame_index"
                ],
                timestamp=point[
                    "timestamp"
                ],
                x=point["x"],
                y=point["y"],
                width=point[
                    "width"
                ],
                height=point[
                    "height"
                ],
            )
            for point in raw["points"]
        ]

        tracks.append(
            ObjectTrack(
                track_id=raw[
                    "track_id"
                ],
                video_id=raw[
                    "video_id"
                ],
                label=raw[
                    "label"
                ],
                start_frame=raw[
                    "start_frame"
                ],
                end_frame=raw[
                    "end_frame"
                ],
                start_time=raw[
                    "start_time"
                ],
                end_time=raw[
                    "end_time"
                ],
                points=points,
            )
        )

    return tracks


def get_point(
    track: ObjectTrack,
    frame_index: int,
):

    for point in track.points:

        if (
            point.frame_index
            ==
            frame_index
        ):
            return point

    return None


def draw_track(
    frame,
    track: ObjectTrack,
    frame_index: int,
):

    point = get_point(
        track,
        frame_index,
    )

    if point is None:
        return

    print(
        f"{track.label} #{track.track_id}: "
        f"x={point.x}, y={point.y}, "
        f"w={point.width}, h={point.height}"
    )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    normalized = (
        0.0 <= point.x <= 1.0
        and 0.0 <= point.y <= 1.0
        and 0.0 <= point.width <= 1.0
        and 0.0 <= point.height <= 1.0
    )

    if normalized:
        x1 = int(point.x * frame_width)
        y1 = int(point.y * frame_height)
        box_width = int(point.width * frame_width)
        box_height = int(point.height * frame_height)
    else:
        x1 = int(point.x)
        y1 = int(point.y)
        box_width = int(point.width)
        box_height = int(point.height)

    x2 = x1 + box_width
    y2 = y1 + box_height

    cx = x1 + box_width // 2
    cy = y1 + box_height // 2

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.circle(
        frame,
        (cx, cy),
        5,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        frame,
        f"{track.label} #{track.track_id}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"center=({cx},{cy})",
        (x1, min(frame_height - 10, y2 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
    )

    parser.add_argument(
        "subject_file",
    )

    parser.add_argument(
        "object_file",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        default=(
            "spatial_debug.jpg"
        ),
    )

    args = parser.parse_args()

    subject_tracks = load_tracks(
        args.subject_file
    )

    object_tracks = load_tracks(
        args.object_file
    )

    capture = cv2.VideoCapture(
        args.video
    )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        args.frame,
    )

    success, frame = (
        capture.read()
    )

    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame "
            f"{args.frame}"
        )

    for track in subject_tracks:

        draw_track(
            frame,
            track,
            args.frame,
        )

    for track in object_tracks:

        draw_track(
            frame,
            track,
            args.frame,
        )

    cv2.imwrite(
        args.output,
        frame,
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()