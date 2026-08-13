from src.storage.track_store import (
    TrackStore,
)

from src.storage.frame_metadata_lookup import (
    FrameMetadataLookup,
)

from src.retrieval.segment_search_engine import (
    SegmentSearchEngine,
)

from src.tracking.sam3_video_tracker import (
    Sam3VideoTracker,
)

from src.tracking.track_builder import (
    TrackBuilder,
)


class ConceptTrackManager:

    def __init__(
        self,
        video_id: str,
        metadata_dir: str = "data/metadata",
        track_dir: str = "data/tracks",
        retrieval_device: str = "cpu",
    ):

        self.video_id = video_id

        self.frame_lookup = (
            FrameMetadataLookup(
                video_id=video_id,
                metadata_dir=
                    metadata_dir,
            )
        )

        self.store = TrackStore(
            root_dir=track_dir
        )

        self.retrieval_device = (
            retrieval_device
        )

        # ----------------------------------
        # IMPORTANT:
        #
        # Don't load CLIP / SAM 3 here.
        #
        # Cache hits should be cheap.
        # ----------------------------------

        self._retriever = None
        self._builder = None

    # ======================================
    # Lazy model loading
    # ======================================

    def _get_retriever(
        self,
    ) -> SegmentSearchEngine:

        if self._retriever is None:

            print(
                "\nLoading V0.2 "
                "retrieval engine..."
            )

            self._retriever = (
                SegmentSearchEngine(
                    video_id=
                        self.video_id,

                    device=
                        self.retrieval_device,
                )
            )

        return self._retriever

    def _get_builder(
        self,
    ) -> TrackBuilder:

        if self._builder is None:

            print(
                "\nLoading SAM 3 "
                "video tracker..."
            )

            tracker = (
                Sam3VideoTracker()
            )

            self._builder = (
                TrackBuilder(
                    tracker=tracker
                )
            )

        return self._builder

    # ======================================
    # Generate possible SAM prompt frames
    # ======================================

    def _candidate_prompt_frames(
        self,
        candidates: list[dict],
        max_attempts: int = 8,
    ) -> list[int]:

        prompt_frames = []

        seen = set()

        for candidate in candidates:

            sampled_frame_ids = (
                candidate.get(
                    "frame_ids",
                    [],
                )
            )

            if not sampled_frame_ids:
                continue

            count = len(
                sampled_frame_ids
            )

            # Prefer middle first.
            selected_positions = [
                count // 2,
                0,
                count - 1,
            ]

            for position in (
                selected_positions
            ):

                sampled_frame_id = int(
                    sampled_frame_ids[
                        position
                    ]
                )

                actual_frame = (
                    self.frame_lookup
                    .get_video_frame_number(
                        sampled_frame_id
                    )
                )

                if actual_frame in seen:
                    continue

                seen.add(
                    actual_frame
                )

                prompt_frames.append(
                    actual_frame
                )

                if (
                    len(prompt_frames)
                    >= max_attempts
                ):

                    return prompt_frames

        return prompt_frames

    # ======================================
    # Main entry point
    # ======================================

    def get_or_build(
        self,
        query: str,
        concept: str | None = None,

        candidate_k: int = 5,

        output_prob_thresh:
            float = 0.5,

        max_frames: int | None = None,

        max_prompt_attempts: int = 8,

        force_rebuild: bool = False,
    ) -> dict:

        concept = (
            concept.strip()

            if concept is not None

            else query.strip()
        )

        # ==================================
        # CACHE HIT
        # ==================================

        if (
            not force_rebuild
            and
            self.store.exists(
                video_id=
                    self.video_id,

                label=
                    concept,
            )
        ):

            print(
                f'\nTrack cache hit: '
                f'"{concept}"'
            )

            cached = (
                self.store.load(
                    video_id=
                        self.video_id,

                    label=
                        concept,
                )
            )

            return {
                "cache_hit":
                    True,

                "concept":
                    concept,

                "data":
                    cached,
            }

        # ==================================
        # CACHE MISS
        # ==================================

        print(
            f'\nTrack cache miss: '
            f'"{concept}"'
        )

        retriever = (
            self._get_retriever()
        )

        print(
            "\nFinding likely "
            "seed segments..."
        )

        # Broad thresholds intentionally:
        #
        # retrieval is only choosing
        # SAM seed candidates.

        candidates = (
            retriever.search(
                query=query,

                k=candidate_k,

                min_visual_score=
                    -1.0,

                min_caption_score=
                    -1.0,
            )
        )

        prompt_frames = (
            self._candidate_prompt_frames(
                candidates=
                    candidates,

                max_attempts=
                    max_prompt_attempts,
            )
        )

        print(
            f"Candidate SAM "
            f"prompt frames: "
            f"{prompt_frames}"
        )

        builder = (
            self._get_builder()
        )

        successful_prompt_frame = (
            None
        )

        tracks = []

        # ==================================
        # Try SAM at likely frames until
        # the concept is actually grounded.
        # ==================================

        for (
            attempt_number,
            prompt_frame,
        ) in enumerate(
            prompt_frames,
            start=1,
        ):

            print(
                f"\nSAM seed attempt "
                f"{attempt_number}/"
                f"{len(prompt_frames)}"
            )

            print(
                f"Frame: "
                f"{prompt_frame}"
            )

            candidate_tracks = (
                builder.build_tracks(
                    video_path=
                        self.frame_lookup
                        .video_path,

                    video_id=
                        self.video_id,

                    prompt=
                        concept,

                    fps=
                        self.frame_lookup
                        .fps,

                    prompt_frame=
                        prompt_frame,

                    output_prob_thresh=
                        output_prob_thresh,

                    max_frames=
                        max_frames,

                    direction=
                        "both",
                )
            )

            if not candidate_tracks:

                continue

            tracks = (
                candidate_tracks
            )

            successful_prompt_frame = (
                prompt_frame
            )

            break

        # ==================================
        # Store even a negative result.
        #
        # --force-rebuild lets us retry
        # later if models/settings change.
        # ==================================

        build_info = {
            "query":
                query,

            "concept":
                concept,

            "candidate_k":
                candidate_k,

            "output_prob_thresh":
                output_prob_thresh,

            "max_frames":
                max_frames,

            "prompt_frames_tried":
                prompt_frames,

            "successful_prompt_frame":
                successful_prompt_frame,

            "status":
                (
                    "found"

                    if tracks

                    else "no_match"
                ),
        }

        output_path = (
            self.store.save(
                video_id=
                    self.video_id,

                label=
                    concept,

                video_path=
                    self.frame_lookup
                    .video_path,

                fps=
                    self.frame_lookup
                    .fps,

                tracks=
                    tracks,

                build_info=
                    build_info,
            )
        )

        print(
            f"\nTrack cache saved:"
            f"\n{output_path}"
        )

        data = self.store.load(
            video_id=
                self.video_id,

            label=
                concept,
        )

        return {
            "cache_hit":
                False,

            "concept":
                concept,

            "data":
                data,
        }