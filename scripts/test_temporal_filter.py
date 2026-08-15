from dataclasses import dataclass

from src.query.temporal_filter import (
    TemporalFilter,
)


@dataclass
class TestWindow:

    start_time: float
    end_time: float


def print_result(
    name,
    result,
):

    print(
        "\n============================"
    )

    print(name)

    print(
        "============================"
    )

    print(
        f"Status: {result.status}"
    )

    print(
        f"Confidence: "
        f"{result.confidence:.3f}"
    )

    print(
        f"Gap: "
        f"{result.gap_seconds}"
    )

    print(
        f"Reason: "
        f"{result.reason}"
    )


def main():

    temporal_filter = (
        TemporalFilter()
    )

    # ------------------------------------------
    # Example timings from your video.
    #
    # Change these to roughly match
    # the actual video.
    # ------------------------------------------

    door_opening = TestWindow(
        start_time=2.0,
        end_time=3.5,
    )

    man_walks_through = TestWindow(
        start_time=5.0,
        end_time=7.0,
    )

    # ==========================================
    # DOOR OPENS BEFORE MAN WALKS THROUGH
    # ==========================================

    result = (
        temporal_filter.evaluate(
            first=door_opening,
            second=man_walks_through,
            relation="before",
        )
    )

    print_result(
        "DOOR OPENING BEFORE MAN WALKING",
        result,
    )

    # ==========================================
    # INVERSE SHOULD FAIL
    # ==========================================

    result = (
        temporal_filter.evaluate(
            first=door_opening,
            second=man_walks_through,
            relation="after",
        )
    )

    print_result(
        "DOOR OPENING AFTER MAN WALKING",
        result,
    )

    # ==========================================
    # MAN WALKS AFTER DOOR OPENS
    # ==========================================

    result = (
        temporal_filter.evaluate(
            first=man_walks_through,
            second=door_opening,
            relation="after",
        )
    )

    print_result(
        "MAN WALKING AFTER DOOR OPENING",
        result,
    )


if __name__ == "__main__":
    main()