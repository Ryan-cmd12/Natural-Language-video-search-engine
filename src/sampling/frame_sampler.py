from pathlib import Path

import cv2
from tqdm import tqdm

from src.models.frame import VideoFrame


def sample_frames(
    video_path: str,
    output_dir: str,
    video_id: str,
    sample_fps: float = 1.0,
) -> list[VideoFrame]:

    if sample_fps <= 0:
        raise ValueError(
            "sample_fps must be greater than 0"
        )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    video_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if video_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video FPS: {video_fps}"
        )

    # Example:
    #
    # video = 30 FPS
    # desired sample = 1 FPS
    #
    # keep every 30th frame
    frame_interval = max(
        1,
        int(round(video_fps / sample_fps))
    )

    output_path = Path(output_dir) / video_id
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    sampled_frames: list[VideoFrame] = []

    frame_number = 0
    frame_id = 0

    progress = tqdm(
        total=total_frames,
        desc="Sampling video",
        unit="frame",
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            timestamp = (
                frame_number / video_fps
            )

            filename = (
                f"frame_{frame_id:06d}.jpg"
            )

            frame_path = (
                output_path / filename
            )

            saved = cv2.imwrite(
                str(frame_path),
                frame,
            )

            if not saved:
                cap.release()
                raise RuntimeError(
                    f"Could not save frame "
                    f"{frame_number}"
                )

            sampled_frames.append(
                VideoFrame(
                    frame_id=frame_id,
                    video_id=video_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    image_path=str(frame_path),
                )
            )

            frame_id += 1

        frame_number += 1
        progress.update(1)

    progress.close()
    cap.release()

    return sampled_frames