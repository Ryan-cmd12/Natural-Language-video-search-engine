import argparse
import torch
from src.retrieval.verified_search_engine import (
    VerifiedVideoSearchEngine,
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
        "query"
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
        "--min-confidence",
        type=float,
        default=0.70,
    )

    args = parser.parse_args()

    print(
        "\nLoading verified "
        "video search engine..."
    )

    engine = (
        VerifiedVideoSearchEngine(
            video_id=args.video_id,

            retrieval_device="cuda" if torch.cuda.is_available() else "cpu",
            verifier_device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    print(
        f'\nQuery: "{args.query}"'
    )

    print(
        "\nRetrieving candidates..."
    )

    response = engine.search(
        query=args.query,
        k=args.k,
        candidate_k=args.candidates,

        min_verifier_confidence=
            args.min_confidence,
    )

    matches = response[
        "matches"
    ]

    rejected = response[
        "rejected"
    ]

    # ==================================
    # VERIFIED MATCHES
    # ==================================

    print(
        "\n=============================="
    )

    print(
        "VERIFIED RESULTS"
    )

    print(
        "=============================="
    )

    if not matches:

        print(
            "\nNo verified matches found."
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
            f"Visual score: "
            f"{result['visual_score']:.4f}"
        )

        print(
            f"Caption score: "
            f"{result['caption_score']:.4f}"
        )

        print(
            f"Caption: "
            f"{result['caption']}"
        )

        print(
            f"VLM confidence: "
            f"{result['verifier_confidence']:.2f}"
        )

        print(
            f"Evidence frames: "
            f"{result['evidence_frames']}"
        )

        print(
            f"Reason: "
            f"{result['verifier_reason']}"
        )

    # ==================================
    # REJECTED CANDIDATES
    #
    # Keep this visible while we're
    # developing. It is extremely useful.
    # ==================================

    print(
        "\n=============================="
    )

    print(
        "REJECTED CANDIDATES"
    )

    print(
        "=============================="
    )

    if not rejected:

        print(
            "\nNone."
        )

    for result in rejected:

        start = format_timestamp(
            result["start_time"]
        )

        end = format_timestamp(
            result["end_time"]
        )

        print(
            f"\n{start} -> {end}"
        )

        print(
            f"Visual: "
            f"{result['visual_score']:.4f}"
        )

        print(
            f"Caption: "
            f"{result['caption_score']:.4f}"
        )

        print(
            f"Caption text: "
            f"{result['caption']}"
        )

        print(
            f"VLM match: "
            f"{result['verified_match']}"
        )

        print(
            f"VLM confidence: "
            f"{result['verifier_confidence']:.2f}"
        )

        print(
            f"Reason: "
            f"{result['verifier_reason']}"
        )


if __name__ == "__main__":
    main()