import argparse
from time import perf_counter

from pathlib import Path

from src.ingestion.video_reader import (
    get_video_metadata,
)

from src.storage.track_store import (
    TrackStore,
)

from src.tracking.sam3_video_tracker import (
    Sam3VideoTracker,
)

from src.tracking.track_builder import (
    TrackBuilder,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str,
    )

    parser.add_argument(
        "prompt",
        type=str,
    )

    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--scan-interval",
        type=int,
        default=None,
        help="Search for the prompt every N frames until an object is found.",
    )

    parser.add_argument(
        "--direction",
        type=str,
        choices=[
            "forward",
            "backward",
            "both",
        ],
        default="both",
    )

    args = parser.parse_args()

    t0 = perf_counter()

    metadata = (
        get_video_metadata(
            args.video
        )
    )

    print(
    f"Metadata: "
    f"{perf_counter() - t0:.2f}s"
    )

    total_frames = max(
        1,
        int(
            round(
                metadata.duration
                * metadata.fps
            )
        ),
    )



    print(
        "\n============================"
    )

    print(
        "SAM 3 TRACK INDEX"
    )

    print(
        "============================"
    )

    print(
        f"\nVideo: "
        f"{metadata.video_id}"
    )

    print(
        f"FPS: "
        f"{metadata.fps:.2f}"
    )

    print(
        f"Duration: "
        f"{metadata.duration:.2f}s"
    )

    print(
        f"Prompt: "
        f"{args.prompt}"
    )

    t0 = perf_counter()

    tracker = (
        Sam3VideoTracker()
    )
    print(
    f"SAM 3 initialization: "
    f"{perf_counter() - t0:.2f}s"
    )

    builder = TrackBuilder(
        tracker=tracker
    )

    t0 = perf_counter()

        
    tracks = (
        builder.build_tracks(
            video_path=
                args.video,

            video_id=
                metadata.video_id,

            prompt=
                args.prompt,

            fps=
                metadata.fps,

            prompt_frame=
                args.prompt_frame,

            output_prob_thresh=
                args.threshold,

            max_frames=
                args.max_frames,

            total_frames=
                total_frames,

            scan_interval=
                args.scan_interval,

            direction=
                args.direction,
        )
    )
    print(
    f"Track building: "
    f"{perf_counter() - t0:.2f}s"
    )

    print(
        "\n============================"
    )

    print(
        "TRACKS"
    )

    print(
        "============================"
    )

    if not tracks:

        print(
            "\nNo tracks found."
        )

    for track in tracks:

        print(
            f"\nTrack {track.track_id}"
        )

        print(
            f"Label: "
            f"{track.label}"
        )

        print(
            f"Frames: "
            f"{track.start_frame}"
            f" -> "
            f"{track.end_frame}"
        )

        print(
            f"Time: "
            f"{track.start_time:.2f}"
            f"s -> "
            f"{track.end_time:.2f}s"
        )

        print(
            f"Points: "
            f"{len(track.points)}"
        )

    t0 = perf_counter()

    store = TrackStore()

    print(
    f"Saving index: "
    f"{perf_counter() - t0:.2f}s"
    )

    output = store.save(
        video_id=
            metadata.video_id,

        label=
            args.prompt,

        video_path=
            args.video,

        fps=
            metadata.fps,

        tracks=
            tracks,
    )

    print(
        f"\nTrack index saved:"
        f"\n{output}"
    )


if __name__ == "__main__":
    main()