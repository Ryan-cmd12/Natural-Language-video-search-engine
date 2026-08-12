import argparse
import torch

from src.retrieval.segment_search_engine import (
    SegmentSearchEngine,
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
        "--min-visual-score",
        type=float,
        default=0.23,
    )

    parser.add_argument(
        "--min-caption-score",
        type=float,
        default=0.40,
    )

    args = parser.parse_args()

    engine = SegmentSearchEngine(
        video_id=args.video_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = engine.search(
        query=args.query,
        k=args.k,
        min_visual_score=
            args.min_visual_score,
        min_caption_score=
            args.min_caption_score,
    )
    
    if not results:

        print("\n--- RESULTS ---")
        print(
            "\nNo sufficiently relevant "
            "moments were found."
        )

        return

    print(
        f'\nQuery: "{args.query}"'
    )

    print(
        "\n--- SEGMENT RESULTS ---"
    )

    for rank, result in enumerate(
        results,
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
            f"{start} → {end}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
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
            "Description: "
            f"{result['caption']}"
        )

        print(
            "Frame: "
            f"{result['representative_frame']}"
        )

        print(
            f"Visual: "
            f"{result['visual_score']:.4f} "
            f"{'✓' if result['visual_match'] else '✗'}"
        )

        print(
            f"Caption: "
            f"{result['caption_score']:.4f} "
            f"{'✓' if result['caption_match'] else '✗'}"
        )

        print(
            f"Description: "
            f"{result['caption']}"
        )


if __name__ == "__main__":
    main()