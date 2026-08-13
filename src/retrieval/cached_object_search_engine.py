from src.tracking.concept_track_manager import (
    ConceptTrackManager,
)


class CachedObjectSearchEngine:

    def __init__(
        self,
        video_id: str,
        retrieval_device: str = "cpu",
    ):

        self.video_id = video_id

        self.track_manager = (
            ConceptTrackManager(
                video_id=video_id,

                retrieval_device=
                    retrieval_device,
            )
        )

    def search(
        self,
        query: str,
        concept: str | None = None,
        k: int = 5,

        force_rebuild: bool = False,

        output_prob_thresh:
            float = 0.5,

        candidate_k: int = 5,

        max_frames: int | None = None,
    ) -> dict:

        track_response = (
            self.track_manager
            .get_or_build(
                query=query,

                concept=concept,

                candidate_k=
                    candidate_k,

                output_prob_thresh=
                    output_prob_thresh,

                max_frames=
                    max_frames,

                force_rebuild=
                    force_rebuild,
            )
        )

        track_data = (
            track_response[
                "data"
            ]
        )

        tracks = (
            track_data[
                "tracks"
            ]
        )

        # For now:
        #
        # chronological object appearances.
        #
        # Later this becomes semantic /
        # temporal ranking.

        tracks = sorted(
            tracks,

            key=lambda track:
                track["start_time"],
        )

        return {
            "video_id":
                self.video_id,

            "query":
                query,

            "concept":
                track_response[
                    "concept"
                ],

            "cache_hit":
                track_response[
                    "cache_hit"
                ],

            "track_count":
                len(tracks),

            "tracks":
                tracks[:k],

            "build_info":
                track_data.get(
                    "build_info",
                    {},
                ),
        }