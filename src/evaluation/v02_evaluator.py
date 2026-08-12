from dataclasses import dataclass


@dataclass
class EvaluationCounts:
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0


def temporal_iou(
    predicted_start: float,
    predicted_end: float,
    true_start: float,
    true_end: float,
) -> float:

    intersection_start = max(
        predicted_start,
        true_start,
    )

    intersection_end = min(
        predicted_end,
        true_end,
    )

    intersection = max(
        0.0,
        intersection_end - intersection_start,
    )

    union_start = min(
        predicted_start,
        true_start,
    )

    union_end = max(
        predicted_end,
        true_end,
    )

    union = (
        union_end - union_start
    )

    if union <= 0:
        return 0.0

    return intersection / union


def best_temporal_iou(
    matches: list[dict],
    ground_truth: list[dict],
) -> float:

    best_iou = 0.0

    for match in matches:

        for truth in ground_truth:

            iou = temporal_iou(
                predicted_start=
                    match["start_time"],

                predicted_end=
                    match["end_time"],

                true_start=
                    truth["start"],

                true_end=
                    truth["end"],
            )

            best_iou = max(
                best_iou,
                iou,
            )

    return best_iou


def calculate_metrics(
    counts: EvaluationCounts,
) -> dict:

    tp = counts.true_positive
    fp = counts.false_positive
    tn = counts.true_negative
    fn = counts.false_negative

    total = tp + fp + tn + fn

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate":
            false_positive_rate,
    }