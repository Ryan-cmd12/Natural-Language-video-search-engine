import argparse
import json
from dataclasses import asdict
from pathlib import Path
import torch

from src.ingestion.video_reader import (
    get_video_metadata,
)

from src.sampling.frame_sampler import (
    sample_frames,
)

from src.embeddings.clip_embedder import (
    CLIPEmbedder,
)

from src.indexing.faiss_indexing import (
    FaissIndex,
)


FRAME_DIR = Path("data/frames")
INDEX_DIR = Path("data/indexes")
METADATA_DIR = Path("data/metadata")


def index_video(
    video_path: str,
    sample_fps: float = 1.0,
):

    print("\n--- VIDEO METADATA ---")

    metadata = get_video_metadata(
        video_path
    )

    print(
        f"Video ID: {metadata.video_id}"
    )

    print(
        f"FPS: {metadata.fps:.2f}"
    )

    print(
        f"Frames: {metadata.frame_count}"
    )

    print(
        f"Duration: {metadata.duration:.2f}s"
    )

    print(
        f"Resolution: "
        f"{metadata.width}x{metadata.height}"
    )

    print("\n--- SAMPLING FRAMES ---")

    frames = sample_frames(
        video_path=video_path,
        output_dir=str(FRAME_DIR),
        video_id=metadata.video_id,
        sample_fps=sample_fps,
    )

    print(
        f"\nSampled {len(frames)} frames."
    )

    image_paths = [
        frame.image_path
        for frame in frames
    ]

    print("\n--- GENERATING EMBEDDINGS ---")

    embedder = CLIPEmbedder(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    embeddings = (
        embedder.encode_images(
            image_paths,
            batch_size=16,
        )
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    dimension = embeddings.shape[1]

    print("\n--- BUILDING FAISS INDEX ---")

    faiss_index = FaissIndex(
        dimension=dimension
    )

    faiss_index.add(
        embeddings
    )

    index_path = (
        INDEX_DIR /
        f"{metadata.video_id}.faiss"
    )

    faiss_index.save(
        str(index_path)
    )

    print(
        f"Index saved to: {index_path}"
    )

    metadata_path = (
        METADATA_DIR /
        f"{metadata.video_id}.json"
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "video": asdict(metadata),
        "sample_fps": sample_fps,
        "embedding_dimension": dimension,
        "frame_count": len(frames),
        "frames": [
            frame.to_dict()
            for frame in frames
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
        )

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )

    print("\nIndexing complete.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "video",
        type=str,
        help="Path to video file",
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    index_video(
        video_path=args.video,
        sample_fps=args.sample_fps,
    )


if __name__ == "__main__":
    main()