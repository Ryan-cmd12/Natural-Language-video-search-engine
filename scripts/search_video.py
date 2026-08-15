import argparse

import json

import gc

import torch

from pathlib import Path
import subprocess
import sys

from src.query.entity_resolver import (
    EntityResolver,
)

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

from src.reasoning.qwen_verifier import (
    QwenVideoVerifier,
)

from src.query.qwen_attribute_verifier import (
    QwenAttributeVerifier
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


def resolve_video(
    video_arg: str,
    track_store: TrackStore,):

    path = Path(
        video_arg
    )

    #
    # Raw/local video supplied.
    #

    if path.exists():

        video_path = str(
            path.resolve()
        )

        video_id = (
            path.stem
        )

        return (
            video_id,
            video_path,
        )

    #
    # Otherwise treat the argument
    # as an existing video_id.
    #

    video_id = (
        video_arg
    )

    video_path = (
        track_store.get_video_path(
            video_id
        )
    )

    if not video_path:

        raise FileNotFoundError(
            f'Video "{video_arg}" is '
            f"not indexed and is not "
            f"a valid file path."
        )

    return (
        video_id,
        video_path,
    )



def ensure_entity_indexes(
    compiled_query,
    video_id: str,
    video_path: str,
    track_store: TrackStore,
) -> TrackStore:

    store = (
        track_store
    )

    for entity in (
        compiled_query.entities
    ):

        label = (
            entity.concept
        )

        print(
            f'\nChecking index for '
            f'"{label}"...'
        )

        #
        # Important:
        #
        # Resolve only the base concept here.
        #
        # We intentionally DO NOT use:
        #
        #   red van
        #   large dog
        #
        # as SAM prompts.
        #
        # Attributes are handled later by
        # AttributeFilter/Qwen.
        #

        resolver = (
            EntityResolver(
                track_store=store
            )
        )

        resolved = (
            resolver.resolve(
                video_id=video_id,

                entity_id=
                    entity.id,

                label=
                    label,

                attributes={},
            )
        )

        if resolved.found:

            print(
                f'  Existing compatible '
                f'index found for "{label}".'
            )

            continue

        # =====================================
        # MISSING INDEX
        # =====================================

        print(
            f'  No index found for '
            f'"{label}".'
        )

        print(
            f'  Building SAM3 index...'
        )

        command = [
            sys.executable,
            "-m",
            "scripts.index_tracks",
            video_path,
            label,
        ]

        subprocess.run(
            command,
            check=True,
        )

        #
        # Reload the TrackStore.
        #
        # This protects us if TrackStore
        # caches its index manifest when
        # constructed.
        #

        store = (
            TrackStore()
        )

        print(
            f'  Finished indexing '
            f'"{label}".'
        )

    return store


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str,
        help=(
            "Video ID or path to "
            "a local video file."
        ),
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

    parser.add_argument(
        "--show-compiled",
        action="store_true",
    )

    args = parser.parse_args()

    store = (
        TrackStore()
    )

    video_id, video_path = (
        resolve_video(
            video_arg=args.video,
            track_store=store,
        )
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
        f"\nVideo ID: "
        f"{video_id}"
    )

    print(
        f"Video path: "
        f"{video_path}"
    )

    print(
        f"Query: "
        f"{args.query}"
    )

    # -----------------------------------
    # QUERY COMPILATION
    # -----------------------------------

    print(
        "\nCompiling query..."
    )

    compiler = (
        QwenQueryCompiler(device = "cuda" if torch.cuda.is_available() else "cpu")
    )

    compiled_query = (
        compiler.compile(
            args.query
        )
    )

    #
    # Query compiler is no longer needed.
    # Free GPU memory before SAM3/Qwen VLM.
    #

    del compiler

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
    "\nChecking required "
    "video indexes..."
    )

    store = (
        ensure_entity_indexes(
            compiled_query=
                compiled_query,

            video_id=
                video_id,

            video_path=
                video_path,

            track_store=
                store,
        )
    )

    if args.show_compiled:
        print(
            "\n============================"
        )

        print(
            "COMPILED QUERY"
        )

        print(
            "============================"
        )

        print(
            json.dumps(
                compiled_query.to_dict(),
                indent=4,
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

    print("\n============================")
    print("RAW QUERY PLAN")
    print("============================")

    if hasattr(plan, "model_dump"):
        print(
            json.dumps(
                plan.model_dump(),
                indent=4,
                default=str,
            )
        )
    else:
        print(plan)

    # -----------------------------------
    # EXECUTION
    # -----------------------------------

    print(
        "\nExecuting query..."
    )
    qwen_video_verifier = QwenVideoVerifier(
        max_frames=8,
    )

    attribute_verifier = (
        QwenAttributeVerifier(
            ask_fn=
                qwen_video_verifier.ask_image
        )
    )

    executor = QueryExecutor(
        track_store=store,

        attribute_verifier=(
            attribute_verifier
        ),

        video_verifier=(
            qwen_video_verifier
        ),
    )

    results = executor.execute(
        plan=plan,
        video_id=video_id,
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