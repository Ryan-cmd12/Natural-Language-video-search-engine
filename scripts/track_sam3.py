import argparse
import json

from src.tracking.sam3_video_tracker import Sam3VideoTracker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "resource_path",
        type=str,
        help="MP4 file path or JPEG frame folder"
    )

    parser.add_argument(
        "frame_index",
        type=int,
        help="Frame index to place the prompt on"
    )

    parser.add_argument(
        "prompt",
        type=str,
        help='Text prompt, e.g. "bus" or "red backpack"'
    )

    args = parser.parse_args()

    tracker = Sam3VideoTracker()

    session_id = tracker.start_session(
        resource_path=args.resource_path
    )

    outputs = tracker.add_text_prompt(
        session_id=session_id,
        frame_index=args.frame_index,
        text=args.prompt,
    )

    print("\n====================")
    print("SAM 3 TRACKING OUTPUT")
    print("====================")
    print(json.dumps(outputs, indent=2, default=str))


if __name__ == "__main__":
    main()