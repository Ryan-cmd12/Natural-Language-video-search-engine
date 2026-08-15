import argparse
import json

from pathlib import Path

from src.models.tracked_object import (
    ObjectTrack,
    TrackPoint,
)

from src.query.spatial_relationship_filter import (
    SpatialRelationshipFilter,
)


def load_tracks(
    path: str,
) -> list[ObjectTrack]:

    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    # Support a few possible TrackStore
    # JSON layouts.
    if isinstance(data, list):

        raw_tracks = data

    elif (
        isinstance(data, dict)
        and
        "tracks" in data
    ):

        raw_tracks = data["tracks"]

    elif (
        isinstance(data, dict)
        and
        "track_id" in data
        and
        "points" in data
    ):

        raw_tracks = [data]

    else:

        raise ValueError(
            f"Unknown track JSON "
            f"format in {path}"
        )

    tracks = []

    for raw_track in raw_tracks:

        points = [
            TrackPoint(
                frame_index=(
                    point["frame_index"]
                ),
                timestamp=(
                    point["timestamp"]
                ),
                x=(
                    point["x"]
                ),
                y=(
                    point["y"]
                ),
                width=(
                    point["width"]
                ),
                height=(
                    point["height"]
                ),
            )
            for point in raw_track[
                "points"
            ]
        ]

        track = ObjectTrack(
            track_id=(
                raw_track["track_id"]
            ),
            video_id=(
                raw_track["video_id"]
            ),
            label=(
                raw_track["label"]
            ),
            start_frame=(
                raw_track["start_frame"]
            ),
            end_frame=(
                raw_track["end_frame"]
            ),
            start_time=(
                raw_track["start_time"]
            ),
            end_time=(
                raw_track["end_time"]
            ),
            points=points,
        )

        tracks.append(
            track
        )

    return tracks


def print_track_summary(
    name: str,
    tracks: list[ObjectTrack],
):

    print(
        "\n============================"
    )

    print(
        f"{name} TRACKS"
    )

    print(
        "============================"
    )

    print(
        f"Count: {len(tracks)}"
    )

    for track in tracks:

        print(
            f"\nTrack ID: "
            f"{track.track_id}"
        )

        print(
            f"Label: "
            f"{track.label}"
        )

        print(
            f"Video: "
            f"{track.video_id}"
        )

        print(
            f"Frames: "
            f"{track.start_frame}"
            f" -> "
            f"{track.end_frame}"
        )

        print(
            f"Points: "
            f"{len(track.points)}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "subject_file",
        type=str,
    )

    parser.add_argument(
        "object_file",
        type=str,
    )

    parser.add_argument(
        "relationship",
        type=str,
        choices=[
            "left_of",
            "right_of",
            "above",
            "below",
            "near",
            "overlapping",
        ],
    )

    parser.add_argument(
        "--min-match-ratio",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--min-common-frames",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    subject_tracks = load_tracks(
        args.subject_file
    )

    object_tracks = load_tracks(
        args.object_file
    )

    print_track_summary(
        "SUBJECT",
        subject_tracks,
    )

    print_track_summary(
        "OBJECT",
        object_tracks,
    )

    spatial_filter = (
        SpatialRelationshipFilter(
            min_match_ratio=(
                args.min_match_ratio
            ),
            min_common_frames=(
                args.min_common_frames
            ),
        )
    )

    results = []

    print(
        "\n============================"
    )

    print(
        "SPATIAL RESULTS"
    )

    print(
        "============================"
    )

    for subject_track in subject_tracks:

        for object_track in object_tracks:

            if (
                subject_track.video_id
                !=
                object_track.video_id
            ):
                print(
                    "\nSkipping pair:"
                )

                print(
                    "Tracks belong to "
                    "different videos."
                )

                continue

            result = (
                spatial_filter.evaluate(
                    subject_track=(
                        subject_track
                    ),
                    object_track=(
                        object_track
                    ),
                    relationship=(
                        args.relationship
                    ),
                )
            )

            results.append(
                (
                    subject_track,
                    object_track,
                    result,
                )
            )

    results.sort(
        key=lambda item: (
            item[2].confidence
        ),
        reverse=True,
    )

    if not results:

        print(
            "\nNo track pairs "
            "could be evaluated."
        )

        return

    for (
        subject_track,
        object_track,
        result,
    ) in results:

        print(
            "\n----------------------------"
        )

        print(
            f"{subject_track.label} "
            f"#{subject_track.track_id}"
        )

        print(
            f"    {args.relationship}"
        )

        print(
            f"{object_track.label} "
            f"#{object_track.track_id}"
        )

        print()

        print(
            f"Status: "
            f"{result.status}"
        )

        print(
            f"Confidence: "
            f"{result.confidence:.3f}"
        )

        print(
            f"Common frames: "
            f"{len(result.evaluated_frames)}"
        )

        print(
            f"Matching frames: "
            f"{len(result.matching_frames)}"
        )

        if result.evaluated_frames:

            print(
                f"Evaluated range: "
                f"{result.evaluated_frames[0]}"
                f" -> "
                f"{result.evaluated_frames[-1]}"
            )

        if result.matching_frames:

            print(
                f"Matching range: "
                f"{result.matching_frames[0]}"
                f" -> "
                f"{result.matching_frames[-1]}"
            )

        print(
            f"Reason: "
            f"{result.reason}"
        )

    # -------------------------
    # BEST RESULT
    # -------------------------

    best = results[0]

    subject_track = best[0]
    object_track = best[1]
    result = best[2]

    print(
        "\n============================"
    )

    print(
        "BEST RESULT"
    )

    print(
        "============================"
    )

    print(
        f"{subject_track.label} "
        f"#{subject_track.track_id}"
    )

    print(
        args.relationship
    )

    print(
        f"{object_track.label} "
        f"#{object_track.track_id}"
    )

    print(
        f"\nStatus: "
        f"{result.status}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.3f}"
    )


if __name__ == "__main__":
    main()