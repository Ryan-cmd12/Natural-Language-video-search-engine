import argparse

from src.query.entity_resolver import (
    EntityResolver,
)

from src.storage.track_store import (
    TrackStore,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id",
        type=str,
    )

    parser.add_argument(
        "entity",
        type=str,
    )

    args = parser.parse_args()

    store = TrackStore()

    resolver = EntityResolver(
        track_store=store
    )

    result = resolver.resolve(
        video_id=
            args.video_id,

        entity_id=
            "test_entity",

        label=
            args.entity,
    )

    print(
        "\n============================"
    )

    print(
        "ENTITY RESOLUTION"
    )

    print(
        "============================"
    )

    print(
        f"\nRequested: "
        f"{result.requested_label}"
    )

    print(
        f"Found: "
        f"{result.found}"
    )

    for match in result.matches:

        print(
            f"\nMatch: "
            f"{match.index.label}"
        )

        print(
            f"Score: "
            f"{match.score:.3f}"
        )

        print(
            f"Reason: "
            f"{match.reason}"
        )


if __name__ == "__main__":
    main()