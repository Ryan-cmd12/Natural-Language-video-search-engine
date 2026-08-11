import argparse
import json

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.segmentation.fixed_window_segmenter import (
    create_fixed_segments,
)

from src.embeddings.text_embedder import (
    TextEmbedder,
)

from src.embeddings.clip_embedder import (
    CLIPEmbedder,
)

from src.captioning.blip_captioner import (
    BLIPCaptioner,
)

from src.indexing.faiss_indexing import (
    FaissIndex,
)


METADATA_DIR = Path(
    "data/metadata"
)

INDEX_DIR = Path(
    "data/indexes"
)


def normalize(
    vector: np.ndarray,
) -> np.ndarray:

    norm = np.linalg.norm(
        vector
    )

    if norm < 1e-12:
        return vector

    return vector / norm


def index_segments(
    video_id: str,
    segment_seconds: float = 5.0,
):

    # ==========================================
    # Load V0.1 metadata
    # ==========================================

    frame_metadata_path = (
        METADATA_DIR /
        f"{video_id}.json"
    )

    if not frame_metadata_path.exists():
        raise FileNotFoundError(
            f"V0.1 metadata not found: "
            f"{frame_metadata_path}"
        )

    with open(
        frame_metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        frame_metadata = json.load(
            f
        )

    frames = frame_metadata[
        "frames"
    ]

    video_metadata = frame_metadata[
        "video"
    ]

    duration = float(
        video_metadata["duration"]
    )

    print(
        f"\nLoaded {len(frames)} "
        f"V0.1 frames."
    )

    # ==========================================
    # Build 5-second segments
    # ==========================================

    print(
        "\n--- CREATING SEGMENTS ---"
    )

    segments = create_fixed_segments(
        frames=frames,
        video_id=video_id,
        video_duration=duration,
        segment_seconds=
            segment_seconds,
    )

    print(
        f"Created {len(segments)} "
        f"segments."
    )

    # ==========================================
    # Load CLIP
    # ==========================================

    clip = CLIPEmbedder(
        device="cpu"
    )

    # ==========================================
    # Embed ALL frames once
    # ==========================================

    print(
        "\n--- EMBEDDING FRAMES ---"
    )

    all_frame_paths = [
        frame["image_path"]
        for frame in frames
    ]

    frame_embeddings = (
        clip.encode_images(
            all_frame_paths,
            batch_size=16,
        )
    )

    # Map:
    #
    # frame_id -> embedding

    embedding_by_frame_id = {}

    for frame, embedding in zip(
        frames,
        frame_embeddings,
    ):

        embedding_by_frame_id[
            int(frame["frame_id"])
        ] = embedding

    # ==========================================
    # Pool frame embeddings into
    # SEGMENT embeddings.
    # ==========================================

    print(
        "\n--- BUILDING SEGMENT "
        "EMBEDDINGS ---"
    )

    segment_embeddings = []

    for segment in segments:

        embeddings = [
            embedding_by_frame_id[
                frame_id
            ]
            for frame_id
            in segment.frame_ids
        ]

        embeddings = np.stack(
            embeddings
        )

        # Mean pooling across time.
        pooled = np.mean(
            embeddings,
            axis=0,
        )

        pooled = normalize(
            pooled
        )

        segment_embeddings.append(
            pooled
        )

    segment_embeddings = (
        np.stack(
            segment_embeddings
        )
        .astype("float32")
    )

    print(
        "Segment embeddings:",
        segment_embeddings.shape,
    )

    # ==========================================
    # Visual FAISS index
    # ==========================================

    visual_index = FaissIndex(
        dimension=
            segment_embeddings.shape[1]
    )

    visual_index.add(
        segment_embeddings
    )

    visual_index_path = (
        INDEX_DIR /
        f"{video_id}_segments_visual.faiss"
    )

    visual_index.save(
        str(visual_index_path)
    )

    # ==========================================
    # Caption representative frame
    # ==========================================

    print(
        "\n--- GENERATING CAPTIONS ---"
    )

    captioner = BLIPCaptioner(
        device="cpu"
    )

    for segment in tqdm(
        segments,
        desc="Captioning segments",
    ):

        caption = captioner.caption(
            segment
            .representative_frame_path
        )
        if not is_valid_caption(
            caption
        ):
            print(
                f"Warning: invalid caption "
                f"for segment "
                f"{segment.segment_id}: "
                f"{caption}"
            )

            caption = ""

        segment.caption = caption

    # ==========================================
    # Caption embeddings
    # ==========================================

    print(
        "\n--- EMBEDDING CAPTIONS ---"
    )

    captions = [
        segment.caption
        for segment in segments
    ]

    # ==========================================
    # Caption text embeddings
    # ==========================================

    print(
        "\n--- EMBEDDING CAPTIONS ---"
    )

    text_embedder = TextEmbedder(
        device="cpu"
    )

    captions = [
        segment.caption
        for segment in segments
    ]

    caption_embeddings = (
        text_embedder.encode(
            captions,
            batch_size=32,
        )
    )

    for i, caption in enumerate(captions):

        if not caption.strip():

            caption_embeddings[
                i
            ] = 0.0

    caption_index = FaissIndex(
        dimension=
            caption_embeddings.shape[1]
    )

    caption_index.add(
        caption_embeddings
    )

    caption_index_path = (
        INDEX_DIR /
        f"{video_id}_segments_caption.faiss"
    )

    caption_index.save(
        str(caption_index_path)
    )

    caption_index = FaissIndex(
        dimension=
            caption_embeddings.shape[1]
    )

    caption_index.add(
        caption_embeddings
    )

    caption_index_path = (
        INDEX_DIR /
        f"{video_id}_segments_caption.faiss"
    )

    caption_index.save(
        str(caption_index_path)
    )

    # ==========================================
    # Save V0.2 metadata
    # ==========================================

    segment_metadata_path = (
        METADATA_DIR /
        f"{video_id}_segments.json"
    )

    output = {
        "video_id": video_id,

        "segment_seconds":
            segment_seconds,

        "segment_count":
            len(segments),

        "segments": [
            segment.to_dict()
            for segment in segments
        ],
    }

    with open(
        segment_metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print(
        "\n--- V0.2 INDEX COMPLETE ---"
    )

    print(
        f"Visual index: "
        f"{visual_index_path}"
    )

    print(
        f"Caption index: "
        f"{caption_index_path}"
    )

    print(
        f"Metadata: "
        f"{segment_metadata_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video_id",
        type=str,
    )

    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=5.0,
    )

    args = parser.parse_args()

    index_segments(
        video_id=args.video_id,
        segment_seconds=
            args.segment_seconds,
    )

def is_valid_caption(
    caption: str,
) -> bool:

    words = (
        caption
        .lower()
        .strip()
        .split()
    )

    if len(words) == 0:
        return False

    # Too much repetition usually
    # indicates generation failure.
    if len(words) >= 6:

        unique_ratio = (
            len(set(words))
            / len(words)
        )

        if unique_ratio < 0.5:
            return False
    return True

if __name__ == "__main__":
    main()