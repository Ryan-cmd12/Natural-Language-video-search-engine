import argparse

from src.storage.track_store import (
    TrackStore,
)


def format_timestamp(
    seconds: float,
) -> str:

    seconds = int(
        seconds
    )

    hours = (
        seconds // 3600
    )

    minutes = (
        seconds % 3600
    ) // 60

    secs = (
        seconds % 60
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id"
    )

    parser.add_argument(
        "label"
    )

    args = parser.parse_args()

    store = TrackStore()

    data = store.load(
        video_id=
            args.video_id,

        label=
            args.label,
    )

    tracks = data[
        "tracks"
    ]

    print(
        f'\nTracks for '
        f'"{args.label}"'
    )

    print(
        "\n============================"
    )

    if not tracks:

        print(
            "No tracks found."
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
            f"\nTrack "
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
            f"Tracked points: "
            f"{len(track['points'])}"
        )


if __name__ == "__main__":
    main()