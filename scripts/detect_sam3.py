import argparse

from src.detection.sam3_detector import Sam3Detector


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image", type=str)
    parser.add_argument("prompt", type=str)
    parser.add_argument("--min-score", type=float, default=0.3)

    args = parser.parse_args()

    detector = Sam3Detector()

    detections = detector.detect(
        image_path=args.image,
        prompt=args.prompt,
        min_score=args.min_score,
    )

    print("\n====================")
    print("SAM 3 DETECTIONS")
    print("====================")

    if not detections:
        print("\nNo detections found.")
        return

    for i, det in enumerate(detections, start=1):
        print(f"\n#{i}")
        print(f"Label: {det.label}")
        print(f"Score: {det.score:.4f}")
        print(f"Box: {det.bbox}")


if __name__ == "__main__":
    main()