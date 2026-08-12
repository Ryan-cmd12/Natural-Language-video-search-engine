from src.retrieval.segment_search_engine import (
    SegmentSearchEngine,
)

from src.detection.sam3_detector import (
    Sam3Detector,
)

from src.detection.sam3_segment_grounder import (
    Sam3SegmentGrounder,
)

from src.reasoning.qwen_verifier import (
    QwenVideoVerifier,
)

import torch


class ObjectAwareSearchEngine:

    def __init__(
        self,
        sam_max_frames,
        video_id: str,

        retrieval_device: str = "cpu",

        sam_min_score: float = 0.30,
    ):  
        retrieval_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.video_id = video_id

        print(
            "\nLoading retrieval engine..."
        )

        self.retriever = (
            SegmentSearchEngine(
                video_id=video_id,
                device=retrieval_device,
            )
        )

        print(
            "\nLoading SAM 3..."
        )

        self.sam_detector = (
            Sam3Detector()
        )

        self.grounder = (
            Sam3SegmentGrounder(
                detector=
                    self.sam_detector,

                max_frames=
                    sam_max_frames,

                min_score=
                    sam_min_score,
            )
        )

        print(
            "\nLoading Qwen verifier..."
        )
        
        self.verifier = (
            QwenVideoVerifier(
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_frames = int(sam_max_frames),
            )
        )

    def search(
        self,
        query: str,

        concept: str | None = None,

        k: int = 5,

        candidate_k: int = 10,

        min_visual_score: float = 0.15,

        min_caption_score: float = 0.25,

        min_verifier_confidence:
            float = 0.70,
    ) -> dict:

        # ----------------------------------
        # For now:
        #
        # query:
        # "person carrying a red backpack"
        #
        # concept:
        # "red backpack"
        #
        # Eventually the query compiler
        # will extract this automatically.
        # ----------------------------------

        concept = (
            concept
            if concept is not None
            else query
        )

        # ==================================
        # Stage 1
        # Cheap V0.2 retrieval
        # ==================================

        candidates = (
            self.retriever.search(
                query=query,

                k=candidate_k,

                min_visual_score=
                    min_visual_score,

                min_caption_score=
                    min_caption_score,
            )
        )

        sam_rejected = []
        qwen_rejected = []
        matches = []

        # ==================================
        # Stage 2
        # SAM 3 grounding
        # ==================================

        for candidate in candidates:

            grounding = (
                self.grounder.ground(
                    segment=candidate,
                    prompt=concept,
                )
            )

            if not grounding.matched:

                sam_rejected.append({
                    **candidate,

                    "rejection_stage":
                        "sam3",

                    "rejection_reason":
                        (
                            f'SAM 3 did not '
                            f'ground "{concept}" '
                            f'in the sampled '
                            f'frames.'
                        ),
                })

                continue

            # ==================================
            # Convert grounded object evidence
            # into serializable data.
            #
            # We don't return the raw masks
            # here yet.
            # ==================================

            object_evidence = []

            for grounded_frame in (
                grounding.grounded_frames
            ):

                for detection in (
                    grounded_frame.detections
                ):

                    object_evidence.append({
                        "frame_id":
                            grounded_frame.frame_id,

                        "segment_frame_index":
                            grounded_frame
                            .segment_frame_index,

                        "frame_path":
                            grounded_frame
                            .frame_path,

                        "label":
                            detection.label,

                        "score":
                            detection.score,

                        "bbox":
                            detection.bbox,
                    })

            grounded_candidate = {
                **candidate,

                "sam_prompt":
                    concept,

                "sam_best_score":
                    grounding.best_score,

                "sam_frames_checked":
                    grounding.frames_checked,

                "sam_frames_with_detections":
                    grounding
                    .frames_with_detections,

                "object_evidence":
                    object_evidence,
            }

            # ==================================
            # Stage 3
            # Qwen semantic verification
            # ==================================

            print("\n[QWEN INPUT DEBUG]")
            print("video_id:", self.video_id)
            print("query:", query)

            print("candidate frame paths:")
            for i, evidence in enumerate(
                grounded_candidate.get(
                    "object_evidence", []
                )
            ):
                print(
                    f"  [{i}] "
                    f"{evidence.get('frame_path')}"
                )

            print(
                "candidate keys:",
                grounded_candidate.keys()
            )

            verification = (
                self.verifier.verify(
                    query=query,
                    segment=
                        grounded_candidate,
                )
            )
            passed = (
                verification.match
                and
                verification.confidence
                >=
                min_verifier_confidence
            )

            if not passed:

                qwen_rejected.append({
                    **grounded_candidate,

                    "rejection_stage":
                        "qwen",

                    "verifier_match":
                        verification.match,

                    "verifier_confidence":
                        verification.confidence,

                    "verifier_reason":
                        verification.reason,
                })

                continue

            # ==================================
            # VERIFIED OBJECT-AWARE RESULT
            # ==================================

            matches.append({
                **grounded_candidate,

                "verified_match":
                    True,

                "verifier_confidence":
                    verification.confidence,

                "verifier_reason":
                    verification.reason,

                "evidence_frames":
                    verification
                    .evidence_frames,
            })

            if len(matches) >= k:
                break

        return {
            "video_id":
                self.video_id,

            "query":
                query,

            "concept":
                concept,

            "candidate_count":
                len(candidates),

            "matches":
                matches,

            "sam_rejected":
                sam_rejected,

            "qwen_rejected":
                qwen_rejected,
        }