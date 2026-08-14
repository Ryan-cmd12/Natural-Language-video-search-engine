import argparse

from src.storage.track_store import (
    TrackStore,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id",
        type=str,
    )

    args = parser.parse_args()

    store = TrackStore()

    indexes = (
        store.list_indexes(
            video_id=
                args.video_id
        )
    )

    print(
        "\n============================"
    )

    print(
        "TRACK INDEXES"
    )

    print(
        "============================"
    )

    print(
        f"\nVideo: {args.video_id}"
    )

    if not indexes:

        print(
            "\nNo indexes found."
        )

        return

    for index in indexes:

        print(
            f"\nLabel: "
            f"{index.label}"
        )

        print(
            f"Tracks: "
            f"{index.track_count}"
        )

        print(
            f"FPS: "
            f"{index.fps}"
        )

        print(
            f"Path: "
            f"{index.path}"
        )


if __name__ == "__main__":
    main()