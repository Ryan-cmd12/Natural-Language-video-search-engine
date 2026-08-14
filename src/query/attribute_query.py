from dataclasses import (
    dataclass,
    field,
)

from src.models.tracked_object import (
    ObjectTrack,
)


@dataclass
class AttributeDecision:

    status: str

    # MATCH
    # REJECT
    # UNVERIFIED

    reason: str

    confidence: float = 1.0


@dataclass
class AttributeFilterResult:

    verified: list[ObjectTrack] = field(
        default_factory=list
    )

    rejected: list[ObjectTrack] = field(
        default_factory=list
    )

    unverified: list[ObjectTrack] = field(
        default_factory=list
    )

    reasons: dict[str, str] = field(
        default_factory=dict
    )


class AttributeFilter:

    COLOR_WORDS = {
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
        "grey",
        "gray",
        "brown",
        "silver",
        "gold",
    }

    # ==================================================
    # FILTER MULTIPLE TRACKS
    # ==================================================

    def filter_tracks(
        self,
        tracks: list[ObjectTrack],
        attributes,
    ) -> AttributeFilterResult:

        result = AttributeFilterResult()

        for track in tracks:

            decision = self.evaluate(
                track=track,
                attributes=attributes,
            )

            key = str(
                track.track_id
            )

            result.reasons[key] = (
                decision.reason
            )

            if decision.status == "MATCH":

                result.verified.append(
                    track
                )

            elif decision.status == "REJECT":

                result.rejected.append(
                    track
                )

            else:

                result.unverified.append(
                    track
                )

        return result

    # ==================================================
    # SINGLE TRACK
    # ==================================================

    def evaluate(
        self,
        track: ObjectTrack,
        attributes,
    ) -> AttributeDecision:

        attributes = (
            self._normalise_attributes(
                attributes
            )
        )

        if not attributes:

            return AttributeDecision(
                status="MATCH",
                reason="no_attributes_requested",
            )

        label_tokens = self._tokens(
            track.label
        )

        unresolved = []

        for name, value in (
            attributes.items()
        ):

            decision = (
                self._evaluate_attribute(
                    name=name,
                    value=value,
                    label_tokens=label_tokens,
                )
            )

            if (
                decision.status
                == "REJECT"
            ):

                return decision

            if (
                decision.status
                == "UNVERIFIED"
            ):

                unresolved.append(
                    decision.reason
                )

        if unresolved:

            return AttributeDecision(
                status="UNVERIFIED",
                reason="; ".join(
                    unresolved
                ),
            )

        return AttributeDecision(
            status="MATCH",
            reason="all_attributes_verified",
        )

    # ==================================================
    # ATTRIBUTE TYPES
    # ==================================================

    def _evaluate_attribute(
        self,
        name: str,
        value,
        label_tokens: set[str],
    ) -> AttributeDecision:

        name = (
            str(name)
            .lower()
            .strip()
        )

        if name in {
            "color",
            "colour",
        }:

            return self._evaluate_color(
                value=value,
                label_tokens=label_tokens,
            )

        value_tokens = (
            self._value_tokens(
                value
            )
        )

        if not value_tokens:

            return AttributeDecision(
                status="UNVERIFIED",
                reason=(
                    f"cannot interpret "
                    f"{name}={value}"
                ),
            )

        if value_tokens.issubset(
            label_tokens
        ):

            return AttributeDecision(
                status="MATCH",
                reason=(
                    f"{name} verified "
                    f"from track label"
                ),
            )

        return AttributeDecision(
            status="UNVERIFIED",
            reason=(
                f"{name}={value} "
                f"requires visual verification"
            ),
        )

    def _evaluate_color(
        self,
        value,
        label_tokens: set[str],
    ) -> AttributeDecision:

        requested = (
            self._value_tokens(value)
            &
            self.COLOR_WORDS
        )

        if not requested:

            return AttributeDecision(
                status="UNVERIFIED",
                reason=(
                    f"unknown color: {value}"
                ),
            )

        stored = (
            label_tokens
            &
            self.COLOR_WORDS
        )

        if requested & stored:

            return AttributeDecision(
                status="MATCH",
                reason=(
                    "color verified from "
                    "track label"
                ),
            )

        if stored:

            return AttributeDecision(
                status="REJECT",
                reason=(
                    "conflicting color: "
                    f"{', '.join(stored)}"
                ),
            )

        return AttributeDecision(
            status="UNVERIFIED",
            reason=(
                "color requires visual "
                "verification"
            ),
        )

    # ==================================================
    # HELPERS
    # ==================================================

    def _normalise_attributes(
        self,
        attributes,
    ) -> dict:

        if not attributes:
            return {}

        if isinstance(
            attributes,
            dict,
        ):
            return attributes

        if isinstance(
            attributes,
            list,
        ):

            return {
                f"attribute_{i}": value
                for i, value
                in enumerate(attributes)
            }

        return {
            "attribute": attributes
        }

    def _tokens(
        self,
        text: str,
    ) -> set[str]:

        text = (
            str(text)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )

        return {
            token
            for token
            in text.split()
            if token
        }

    def _value_tokens(
        self,
        value,
    ) -> set[str]:

        if isinstance(
            value,
            (list, tuple, set),
        ):

            result = set()

            for item in value:

                result.update(
                    self._tokens(
                        str(item)
                    )
                )

            return result

        return self._tokens(
            str(value)
        )