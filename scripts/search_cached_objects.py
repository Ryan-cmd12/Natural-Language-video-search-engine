import argparse

from src.retrieval.cached_object_search_engine import (
    CachedObjectSearchEngine,
)


def format_timestamp(
    seconds: float,
) -> str:

    total = int(seconds)

    hours = (
        total // 3600
    )

    minutes = (
        total % 3600
    ) // 60

    seconds = (
        total % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id"
    )

    parser.add_argument(
        "query"
    )

    parser.add_argument(
        "--concept",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    engine = (
        CachedObjectSearchEngine(
            video_id=
                args.video_id,
        )
    )

    response = (
        engine.search(
            query=
                args.query,

            concept=
                args.concept,

            k=
                args.k,

            candidate_k=
                args.candidates,

            output_prob_thresh=
                args.threshold,

            max_frames=
                args.max_frames,

            force_rebuild=
                args.force,
        )
    )

    print(
        "\n============================"
    )

    print(
        "OBJECT TRACK SEARCH"
    )

    print(
        "============================"
    )

    print(
        f"\nQuery: "
        f"{response['query']}"
    )

    print(
        f"Concept: "
        f"{response['concept']}"
    )

    print(
        f"Cache hit: "
        f"{response['cache_hit']}"
    )

    print(
        f"Tracks found: "
        f"{response['track_count']}"
    )

    tracks = response[
        "tracks"
    ]

    if not tracks:

        print(
            "\nNo object tracks found."
        )

        build_info = (
            response.get(
                "build_info",
                {},
            )
        )

        print(
            f"Status: "
            f"{build_info.get('status')}"
        )

        return

    for track in tracks:

        start = format_timestamp(
            track["start_time"]
        )

        end = format_timestamp(
            track["end_time"]
        )

        print(
            "\n----------------------------"
        )

        print(
            f"Track "
            f"{track['track_id']}"
        )

        print(
            f"{start} -> {end}"
        )

        print(
            f"Frames: "
            f"{track['start_frame']}"
            f" -> "
            f"{track['end_frame']}"
        )

        print(
            f"Points: "
            f"{len(track['points'])}"
        )


if __name__ == "__main__":
    main()