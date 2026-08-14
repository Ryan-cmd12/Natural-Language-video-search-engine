from dataclasses import (
    dataclass,
    field,
)

from src.models.tracked_object import (
    ObjectTrack,
)


@dataclass
class CandidateWindow:

    start_time: float
    end_time: float

    tracks: list[ObjectTrack] = field(
        default_factory=list
    )

    score: float = 0.0

    evidence: dict = field(
        default_factory=dict
    )

    def to_dict(self):

        return {
            "start_time":
                self.start_time,

            "end_time":
                self.end_time,

            "score":
                self.score,

            "track_ids": [
                track.track_id
                for track in self.tracks
            ],

            "labels": [
                track.label
                for track in self.tracks
            ],

            "evidence":
                self.evidence,
        }