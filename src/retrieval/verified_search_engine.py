from src.retrieval.segment_search_engine import (
    SegmentSearchEngine,
)

from src.reasoning.qwen_verifier import (
    QwenVideoVerifier,
)


class VerifiedVideoSearchEngine:

    def __init__(
        self,
        video_id: str,
        retrieval_device: str = None,
        verifier_device: str = None,
    ):

        if retrieval_device is None:
            retrieval_device = "cuda" if torch.cuda.is_available() else "cpu"
        if verifier_device is None:
            verifier_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.video_id = video_id

        # Cheap retrieval
        self.retriever = (
            SegmentSearchEngine(
                video_id=video_id,
                device=retrieval_device,
            )
        )

        # Expensive verification
        self.verifier = (
            QwenVideoVerifier(
                device=verifier_device,
            )
        )

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_k: int = 5,
        min_visual_score: float = 0.18,
        min_caption_score: float = 0.30,
        min_verifier_confidence:
            float = 0.70,
    ) -> dict:

        # =================================
        # STEP 1:
        # Broad cheap retrieval
        # =================================

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

        matches = []
        rejected = []

        # =================================
        # STEP 2:
        # Expensive VLM verification
        # =================================

        for candidate in candidates:

            verification = (
                self.verifier.verify(
                    query=query,
                    segment=candidate,
                )
            )

            result = {
                **candidate,

                "verified_match":
                    verification.match,

                "verifier_confidence":
                    verification.confidence,

                "verifier_reason":
                    verification.reason,

                "evidence_frames":
                    verification
                    .evidence_frames,
            }

            # =================================
            # Strict verification gate
            # =================================

            passed = (
                verification.match
                and
                verification.confidence
                >= min_verifier_confidence
            )

            if passed:

                matches.append(
                    result
                )

            else:

                rejected.append(
                    result
                )

            if len(matches) >= k:
                break

        return {
            "query": query,
            "matches": matches,
            "rejected": rejected,
            "candidate_count":
                len(candidates),
        }