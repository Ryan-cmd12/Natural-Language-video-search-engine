import argparse

from src.retrieval.object_aware_search_engine import (
    ObjectAwareSearchEngine,
)


def format_timestamp(
    seconds: float,
) -> str:

    seconds = int(seconds)

    hours = seconds // 3600

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
        "video_id",
    )

    parser.add_argument(
        "query",
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
        default=10,
    )

    parser.add_argument(
        "--sam-score",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--max-frames",
        type = int,
        default = 3
    )

    args = parser.parse_args()

    engine = (
        ObjectAwareSearchEngine(
            video_id=
                args.video_id,

            sam_min_score=
                args.sam_score,

            sam_max_frames = args.max_frames,
        )
    )

    response = engine.search(
        query=args.query,

        concept=args.concept,

        k=args.k,

        candidate_k=
            args.candidates,
    )

    matches = response[
        "matches"
    ]

    print(
        "\n============================"
    )

    print(
        "OBJECT-AWARE RESULTS"
    )

    print(
        "============================"
    )

    if not matches:

        print(
            "\nNo verified "
            "object matches found."
        )

    for rank, result in enumerate(
        matches,
        start=1,
    ):

        start = format_timestamp(
            result["start_time"]
        )

        end = format_timestamp(
            result["end_time"]
        )

        print(
            f"\n#{rank} "
            f"{start} -> {end}"
        )

        print(
            f"SAM prompt: "
            f"{result['sam_prompt']}"
        )

        print(
            f"SAM best score: "
            f"{result['sam_best_score']:.4f}"
        )

        print(
            f"Frames grounded: "
            f"{result['sam_frames_with_detections']}"
            f"/"
            f"{result['sam_frames_checked']}"
        )

        print(
            f"Objects found: "
            f"{len(result['object_evidence'])}"
        )

        for detection in (
            result["object_evidence"]
        ):

            print(
                "\n  Object:"
            )

            print(
                f"    Frame: "
                f"{detection['frame_id']}"
            )

            print(
                f"    Score: "
                f"{detection['score']:.4f}"
            )

            print(
                f"    Box: "
                f"{detection['bbox']}"
            )

        print(
            f"\nVLM confidence: "
            f"{result['verifier_confidence']:.2f}"
        )

        print(
            f"Reason: "
            f"{result['verifier_reason']}"
        )

    print(
        "\n============================"
    )

    print(
        "DEBUG"
    )

    print(
        "============================"
    )

    print(
        f"\nSAM rejected: "
        f"{len(response['sam_rejected'])}"
    )

    print(
        f"Qwen rejected: "
        f"{len(response['qwen_rejected'])}"
    )


if __name__ == "__main__":
    main()