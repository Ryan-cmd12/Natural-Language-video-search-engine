import json

from pathlib import Path

from src.embeddings.clip_embedder import (
    CLIPEmbedder,
)

from src.indexing.faiss_indexing import (
    FaissIndex,
)

from src.embeddings.text_embedder import (
    TextEmbedder,
)

import torch


class SegmentSearchEngine:

    def __init__(
        self,
        video_id: str,
        device: str = None,
    ):  
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        

        self.video_id = video_id

        metadata_path = Path(
            f"data/metadata/"
            f"{video_id}_segments.json"
        )

        visual_index_path = Path(
            f"data/indexes/"
            f"{video_id}_segments_visual.faiss"
        )

        caption_index_path = Path(
            f"data/indexes/"
            f"{video_id}_segments_caption.faiss"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                metadata_path
            )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(
                f
            )

        self.segments = metadata[
            "segments"
        ]

        self.visual_index = (
            FaissIndex.load(
                str(
                    visual_index_path
                )
            )
        )

        self.caption_index = (
            FaissIndex.load(
                str(
                    caption_index_path
                )
            )
        )
        self.clip_embedder = CLIPEmbedder(
            device=device
        )

        self.text_embedder = TextEmbedder(
            device=device
        )

    def search(
        self,
        query: str,
        k: int = 5,
        min_visual_score: float = 0.23,
        min_caption_score: float = 0.40,
    ) -> list[dict]:

        # =====================================
        # Encode the query TWO different ways
        # =====================================

        visual_query = (
            self.clip_embedder.encode_text(
                query
            )
        )

        caption_query = (
            self.text_embedder.encode_query(
                query
            )
        )

        n = len(
            self.segments
        )

        # =====================================
        # IMAGE ↔ TEXT search using CLIP
        # =====================================

        visual_scores, visual_ids = (
            self.visual_index.search(
                visual_query,
                n,
            )
        )

        # =====================================
        # TEXT ↔ TEXT semantic search
        # =====================================

        caption_scores, caption_ids = (
            self.caption_index.search(
                caption_query,
                n,
            )
        )

        # =====================================
        # Build score/rank lookups
        # =====================================

        visual_lookup = {}
        visual_rank = {}

        for rank, (
            score,
            segment_id,
        ) in enumerate(
            zip(
                visual_scores[0],
                visual_ids[0],
            ),
            start=1,
        ):

            if segment_id == -1:
                continue

            segment_id = int(
                segment_id
            )

            visual_lookup[
                segment_id
            ] = float(score)

            visual_rank[
                segment_id
            ] = rank

        caption_lookup = {}
        caption_rank = {}

        for rank, (
            score,
            segment_id,
        ) in enumerate(
            zip(
                caption_scores[0],
                caption_ids[0],
            ),
            start=1,
        ):

            if segment_id == -1:
                continue

            segment_id = int(
                segment_id
            )

            caption_lookup[
                segment_id
            ] = float(score)

            caption_rank[
                segment_id
            ] = rank

        # =====================================
        # Relevance gating
        # =====================================

        candidates = []

        for index, segment in enumerate(
            self.segments
        ):

            visual_score = (
                visual_lookup.get(
                    index,
                    0.0,
                )
            )

            caption_score = (
                caption_lookup.get(
                    index,
                    0.0,
                )
            )

            visual_match = (
                visual_score
                >= min_visual_score
            )

            caption_match = (
                caption_score
                >= min_caption_score
            )

            # ---------------------------------
            # IMPORTANT:
            # Neither channel believes that
            # this segment is relevant.
            # ---------------------------------

            if not (
                visual_match
                or caption_match
            ):
                continue

            # =================================
            # Reciprocal-rank style fusion
            #
            # We're ranking the two channels,
            # NOT directly adding their raw
            # cosine scores.
            # =================================

            visual_position = (
                visual_rank.get(
                    index,
                    n + 1,
                )
            )

            caption_position = (
                caption_rank.get(
                    index,
                    n + 1,
                )
            )

            rank_score = (
                1.0
                / (60 + visual_position)
                +
                1.0
                / (60 + caption_position)
            )

            candidates.append({
                "segment_id":
                    segment[
                        "segment_id"
                    ],

                "start_time":
                    segment[
                        "start_time"
                    ],

                "end_time":
                    segment[
                        "end_time"
                    ],

                "caption":
                    segment[
                        "caption"
                    ],

                "representative_frame":
                    segment[
                        "representative_frame_path"
                    ],

                #For sam
                "frame_ids": segment["frame_ids"],

                #For the verifier, it needs all the frames from a segment
                "frame_paths":
                    segment["frame_paths"],

                "visual_score":
                    visual_score,

                "caption_score":
                    caption_score,

                "visual_match":
                    visual_match,

                "caption_match":
                    caption_match,

                "score":
                    rank_score,
            })

        # =====================================
        # Best evidence first
        # =====================================

        candidates.sort(
            key=lambda result:
                result["score"],
            reverse=True,
        )

        return candidates[:k]