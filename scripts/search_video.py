import argparse

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


def format_timestamp(
    seconds: float,
):

    seconds = max(
        0.0,
        seconds,
    )

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds
        - minutes * 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
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
        "--top-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

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
        f"\nVideo: {args.video_id}"
    )

    print(
        f"Query: {args.query}"
    )

    # -----------------------------------
    # QUERY COMPILATION
    # -----------------------------------

    print(
        "\nCompiling query..."
    )

    compiler = (
        QwenQueryCompiler()
    )

    compiled_query = (
        compiler.compile(
            args.query
        )
    )

    # -----------------------------------
    # PLANNING
    # -----------------------------------

    print(
        "\nBuilding query plan..."
    )

    planner = (
        QueryPlanner()
    )

    plan = planner.plan(
        compiled_query
    )

    # -----------------------------------
    # EXECUTION
    # -----------------------------------

    print(
        "\nExecuting query..."
    )

    store = (
        TrackStore()
    )

    executor = (
        QueryExecutor(
            track_store=store
        )
    )

    results = executor.execute(
        plan=plan,
        video_id=args.video_id,
    )

    # -----------------------------------
    # RESULTS
    # -----------------------------------

    print(
        "\n============================"
    )

    print(
        "SEARCH RESULTS"
    )

    print(
        "============================"
    )

    if not results:

        print(
            "\nNo matching video "
            "segments found."
        )

        return

    for index, result in enumerate(
        results[:args.top_k],
        start=1,
    ):

        start = format_timestamp(
            result.start_time
        )

        end = format_timestamp(
            result.end_time
        )

        labels = sorted(
            set(
                track.label
                for track
                in result.tracks
            )
        )

        print(
            f"\n#{index}"
        )

        print(
            f"Time: {start} -> {end}"
        )

        print(
            f"Score: {result.score:.3f}"
        )

        print(
            "Objects: "
            + ", ".join(labels)
        )

        print(
            "Tracks: "
            + ", ".join(
                str(track.track_id)
                for track
                in result.tracks
            )
        )


if __name__ == "__main__":
    main()