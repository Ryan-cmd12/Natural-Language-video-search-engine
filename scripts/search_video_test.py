import argparse
import json

import torch

from src.query.qwen_query_compiler import (
    QwenQueryCompiler,
)

from src.query.query_planner import (
    QueryPlanner,
)

from src.query.query_executor import (
    QueryExecutor,
)

from src.storage.track_store import (
    TrackStore,
)

from src.query.qwen_video_verifier import (
    QwenVideoVerifier,
)


def format_time(
    seconds: float,
) -> str:

    seconds = max(
        0.0,
        float(seconds),
    )

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        -
        minutes * 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


def print_compiled_query(
    compiled,
):

    print(
        "\n============================"
    )

    print(
        "COMPILED QUERY"
    )

    print(
        "============================\n"
    )

    print(
        json.dumps(
            compiled.to_dict(),
            indent=4,
        )
    )


def print_plan(
    plan,
):

    print(
        "\n============================"
    )

    print(
        "QUERY PLAN"
    )

    print(
        "============================"
    )

    for step in plan.steps:

        print(
            f"\n[{step.step_id}]"
        )

        print(
            f"  operation: "
            f"{step.operation}"
        )

        if step.depends_on:

            print(
                f"  depends on: "
                f"{', '.join(step.depends_on)}"
            )


def print_results(
    query: str,
    video_id: str,
    results,
):

    print(
        "\n============================"
    )

    print(
        "SEARCH RESULTS"
    )

    print(
        "============================"
    )

    print(
        f"\nVideo: {video_id}"
    )

    print(
        f'Query: "{query}"'
    )

    if not results:

        print(
            "\nNo matching "
            "video moments found."
        )

        return

    print(
        f"\nMatches: "
        f"{len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            "\n----------------------------"
        )

        print(
            f"RESULT #{index}"
        )

        print(
            "----------------------------"
        )

        print(
            f"Time: "
            f"{format_time(result.start_time)}"
            f" -> "
            f"{format_time(result.end_time)}"
        )

        print(
            f"Score: "
            f"{result.score:.3f}"
        )

        if result.tracks:

            print(
                "Tracks:"
            )

            for track in (
                result.tracks
            ):

                print(
                    f"  - "
                    f"{track.label}"
                    f" #{track.track_id}"
                )

        #
        # These fields exist only when
        # VLM verification added them.
        #

        if hasattr(
            result,
            "vlm_confidence",
        ):

            print(
                f"VLM confidence: "
                f"{result.vlm_confidence:.3f}"
            )

        if hasattr(
            result,
            "vlm_reason",
        ):

            print(
                f"VLM reason: "
                f"{result.vlm_reason}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id",
        type=str,
    )

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--show-compiled",
        action="store_true",
    )

    parser.add_argument(
        "--show-plan",
        action="store_true",
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n============================"
    )

    print(
        "NATURAL LANGUAGE VIDEO SEARCH"
    )

    print(
        "============================"
    )

    print(
        f"\nVideo: "
        f"{args.video_id}"
    )

    print(
        f'Query: "{args.query}"'
    )

    # ==========================================
    # QUERY COMPILER
    # ==========================================

    compiler = (
        QwenQueryCompiler(
            device=device,
        )
    )

    compiled = (
        compiler.compile(
            args.query
        )
    )

    if args.show_compiled:

        print_compiled_query(
            compiled
        )

    # ==========================================
    # QUERY PLANNER
    # ==========================================

    planner = (
        QueryPlanner()
    )

    try:

        plan = (
            planner.plan(
                compiled
            )
        )

    except NotImplementedError as error:

        print(
            "\nQUERY NOT SUPPORTED"
        )

        print(
            str(error)
        )

        return

    if args.show_plan:

        print_plan(
            plan
        )

    # ==========================================
    # STORAGE
    # ==========================================

    track_store = (
        TrackStore()
    )

    # ==========================================
    # VIDEO VLM
    # ==========================================

    #
    # Used for actions and final candidate
    # verification.
    #

    video_verifier = (
        QwenVideoVerifier(
            max_frames=8,
            device=device,
        )
    )

    # ==========================================
    # ATTRIBUTE VERIFIER
    # ==========================================

    #
    # IMPORTANT:
    #
    # Insert your EXISTING working
    # QwenAttributeVerifier instance here.
    #
    # Do not recreate or change its local
    # path handling.
    #
    # Example:
    #
    # attribute_verifier = (
    #     QwenAttributeVerifier(
    #         ask_fn=your_existing_ask_fn,
    #     )
    # )
    #

    attribute_verifier = None

    # ==========================================
    # EXECUTOR
    # ==========================================

    executor = (
        QueryExecutor(
            track_store=
                track_store,

            attribute_verifier=
                attribute_verifier,

            video_verifier=
                video_verifier,
        )
    )

    # ==========================================
    # EXECUTE
    # ==========================================

    results = (
        executor.execute(
            plan=plan,
            video_id=
                args.video_id,
        )
    )

    # ==========================================
    # DISPLAY
    # ==========================================

    print_results(
        query=args.query,
        video_id=args.video_id,
        results=results,
    )


if __name__ == "__main__":
    main()