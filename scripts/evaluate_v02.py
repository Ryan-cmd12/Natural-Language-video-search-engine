import argparse
import json
import torch
from pathlib import Path

from src.evaluation.v02_evaluator import (
    EvaluationCounts,
    best_temporal_iou,
    calculate_metrics,
)

from src.retrieval.verified_search_engine import (
    VerifiedVideoSearchEngine,
)


def evaluate(
    evaluation_file: str,
    iou_threshold: float = 0.10,
):

    path = Path(
        evaluation_file
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        evaluation_data = json.load(
            f
        )

    video_id = evaluation_data[
        "video_id"
    ]

    queries = evaluation_data[
        "queries"
    ]

    print(
        f"\nEvaluating video: {video_id}"
    )

    print(
        f"Queries: {len(queries)}"
    )

    # ----------------------------------
    # IMPORTANT:
    #
    # Load CLIP / MiniLM / Qwen ONCE.
    # Do not recreate them per query.
    # ----------------------------------

    engine = (
        VerifiedVideoSearchEngine(
            video_id=video_id,

            retrieval_device= "cuda" if torch.cuda.is_available() else "cpu",
            verifier_device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    counts = EvaluationCounts()

    temporal_ious = []

    query_results = []

    for query_number, case in enumerate(
        queries,
        start=1,
    ):

        query = case[
            "query"
        ]

        expected = case[
            "expected"
        ]

        print(
            "\n"
            "================================="
        )

        print(
            f"[{query_number}/{len(queries)}] "
            f"{query}"
        )

        print(
            f"Expected: {expected}"
        )

        response = engine.search(
            query=query,
            k=5,
            candidate_k=5,
        )

        matches = response[
            "matches"
        ]

        # ==================================
        # NEGATIVE QUERY
        # ==================================

        if expected == "no_match":

            if len(matches) == 0:

                counts.true_negative += 1

                status = "PASS"

                print(
                    "Result: NO MATCH"
                )

                print(
                    "PASS - correctly rejected"
                )

            else:

                counts.false_positive += 1

                status = "FAIL"

                print(
                    f"Result: "
                    f"{len(matches)} match(es)"
                )

                print(
                    "FAIL - false positive"
                )

                for match in matches:

                    print(
                        f"  "
                        f"{match['start_time']:.1f}"
                        f" -> "
                        f"{match['end_time']:.1f}"
                    )

                    print(
                        f"  "
                        f"{match['verifier_reason']}"
                    )

            query_results.append({
                "query": query,
                "expected": expected,
                "status": status,
            })

            continue

        # ==================================
        # POSITIVE QUERY
        # ==================================

        ground_truth = case.get(
            "ground_truth",
            [],
        )

        if not matches:

            counts.false_negative += 1

            print(
                "Result: NO MATCH"
            )

            print(
                "FAIL - missed expected event"
            )

            query_results.append({
                "query": query,
                "expected": expected,
                "status": "FAIL",
                "best_iou": 0.0,
            })

            continue

        iou = best_temporal_iou(
            matches=matches,
            ground_truth=ground_truth,
        )

        temporal_ious.append(
            iou
        )

        if iou >= iou_threshold:

            counts.true_positive += 1

            status = "PASS"

            print(
                f"PASS - temporal IoU: "
                f"{iou:.3f}"
            )

        else:

            counts.false_negative += 1

            status = "FAIL"

            print(
                f"FAIL - wrong moment"
            )

            print(
                f"Best temporal IoU: "
                f"{iou:.3f}"
            )

        print(
            "\nReturned matches:"
        )

        for match in matches:

            print(
                f"  "
                f"{match['start_time']:.1f}"
                f" -> "
                f"{match['end_time']:.1f}"
            )

            print(
                f"  VLM: "
                f"{match['verifier_confidence']:.2f}"
            )

            print(
                f"  "
                f"{match['verifier_reason']}"
            )

        query_results.append({
            "query": query,
            "expected": expected,
            "status": status,
            "best_iou": iou,
        })

    # ======================================
    # Final metrics
    # ======================================

    metrics = calculate_metrics(
        counts
    )

    mean_iou = (
        sum(temporal_ious)
        / len(temporal_ious)
        if temporal_ious
        else 0.0
    )

    print(
        "\n\n"
        "================================="
    )

    print(
        "V0.2 EVALUATION RESULTS"
    )

    print(
        "================================="
    )

    print(
        f"True positives:  "
        f"{counts.true_positive}"
    )

    print(
        f"False positives: "
        f"{counts.false_positive}"
    )

    print(
        f"True negatives:  "
        f"{counts.true_negative}"
    )

    print(
        f"False negatives: "
        f"{counts.false_negative}"
    )

    print()

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.3f}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.3f}"
    )

    print(
        f"Recall:              "
        f"{metrics['recall']:.3f}"
    )

    print(
        f"Specificity:         "
        f"{metrics['specificity']:.3f}"
    )

    print(
        f"False positive rate: "
        f"{metrics['false_positive_rate']:.3f}"
    )

    print(
        f"Mean Temporal IoU:   "
        f"{mean_iou:.3f}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "evaluation_file",
        type=str,
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    evaluate(
        evaluation_file=
            args.evaluation_file,

        iou_threshold=
            args.iou_threshold,
    )


if __name__ == "__main__":
    main()