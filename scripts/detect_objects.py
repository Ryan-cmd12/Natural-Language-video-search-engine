import argparse

from src.detection.grounding_dino_detector import (
    GroundingDinoDetector,
)

from src.detection.visualizer import (
    draw_detections,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
    )

    parser.add_argument(
        "labels",
        nargs="+",
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
        #0.4
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        #0.3
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/debug/"
            "detections.jpg"
        ),
    )

    args = parser.parse_args()

    print(
        "\nLoading detector..."
    )

    detector = (
        GroundingDinoDetector(
            device="cpu"
        )
    )

    print(
        "\nSearching for:"
    )

    for label in args.labels:

        print(
            f" - {label}"
        )

    detections = (
        detector.detect(
            image_path=args.image,

            labels=args.labels,

            box_threshold=
                args.box_threshold,

            text_threshold=
                args.text_threshold,
        )
    )

    print(
        "\n======================="
    )

    print(
        "DETECTIONS"
    )

    print(
        "======================="
    )

    if not detections:

        print(
            "\nNo objects detected."
        )

        return

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        print(
            f"\n#{index}"
        )

        print(
            f"Label: "
            f"{detection.label}"
        )

        print(
            f"Score: "
            f"{detection.score:.4f}"
        )

        print(
            f"Box: "
            f"{detection.bbox}"
        )

    draw_detections(
        image_path=
            args.image,

        detections=
            detections,

        output_path=
            args.output,
    )

    print(
        f"\nAnnotated image: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()