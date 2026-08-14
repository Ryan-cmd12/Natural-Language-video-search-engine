from dataclasses import (
    dataclass,
    field,
)

from src.models.track_index import (
    TrackIndexInfo,
)

from src.storage.track_store import (
    TrackStore,
)


@dataclass
class IndexMatch:

    index: TrackIndexInfo

    score: float

    reason: str


@dataclass
class ResolvedEntity:

    entity_id: str

    requested_label: str

    attributes: dict | list = field(
        default_factory=dict
    )

    matches: list[IndexMatch] = field(
        default_factory=list
    )

    @property
    def found(
        self,
    ) -> bool:

        return bool(
            self.matches
        )


class EntityResolver:

    def __init__(
        self,
        track_store: TrackStore,
    ):

        self.track_store = (
            track_store
        )

    # ==================================================
    # PUBLIC API
    # ==================================================

    def resolve(
        self,
        video_id: str,
        entity_id: str,
        label: str,
        attributes=None,
    ) -> ResolvedEntity:

        indexes = (
            self.track_store.list_indexes(
                video_id=video_id
            )
        )

        requested_slug = (
            self.track_store._slugify(
                label
            )
        )

        requested_tokens = (
            self._tokens(
                requested_slug
            )
        )

        matches = []

        for index in indexes:

            stored_slug = (
                self.track_store._slugify(
                    index.label
                )
            )

            stored_tokens = (
                self._tokens(
                    stored_slug
                )
            )

            match = self._match_index(
                requested_slug=
                    requested_slug,

                requested_tokens=
                    requested_tokens,

                stored_slug=
                    stored_slug,

                stored_tokens=
                    stored_tokens,

                index=
                    index,
            )

            if match is not None:

                matches.append(
                    match
                )

        #
        # Best matches first.
        #

        matches.sort(
            key=lambda match:
                match.score,
            reverse=True,
        )

        return ResolvedEntity(
            entity_id=
                entity_id,

            requested_label=
                label,

            attributes=
                attributes or {},

            matches=
                matches,
        )

    # ==================================================
    # MATCHING
    # ==================================================

    def _match_index(
        self,
        requested_slug: str,
        requested_tokens: set[str],
        stored_slug: str,
        stored_tokens: set[str],
        index: TrackIndexInfo,
    ) -> IndexMatch | None:

        #
        # CASE 1
        #
        # Exact:
        #
        # red car -> red car
        #

        if (
            requested_slug
            == stored_slug
        ):

            return IndexMatch(
                index=index,
                score=1.0,
                reason="exact_label",
            )

        #
        # CASE 2
        #
        # Generic query can use a more
        # specific stored index:
        #
        # car -> red car
        # car -> blue car
        #
        # This is SAFE because every
        # red car is still a car.
        #

        if (
            requested_tokens
            and
            requested_tokens.issubset(
                stored_tokens
            )
        ):

            specificity_penalty = (
                len(requested_tokens)
                /
                len(stored_tokens)
            )

            score = (
                0.8
                +
                0.15
                * specificity_penalty
            )

            return IndexMatch(
                index=index,
                score=score,
                reason="stored_index_more_specific",
            )

        #
        # IMPORTANT:
        #
        # We deliberately DON'T do:
        #
        # stored_tokens.issubset(
        #     requested_tokens
        # )
        #
        # yet.
        #
        # Otherwise:
        #
        # query = blue car
        # index = car
        #
        # would be treated as confirmed
        # evidence for a BLUE car.
        #
        # Attribute filtering will handle
        # this later.
        #

        return None

    # ==================================================
    # HELPERS
    # ==================================================

    def _tokens(
        self,
        slug: str,
    ) -> set[str]:

        return {
            token
            for token
            in slug.split("_")
            if token
        }