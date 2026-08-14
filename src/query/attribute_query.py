from dataclasses import dataclass

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

        label_tokens = (
            self._tokens(
                track.label
            )
        )

        unresolved = []

        for (
            attribute_name,
            attribute_value,
        ) in attributes.items():

            decision = (
                self._evaluate_attribute(
                    name=attribute_name,
                    value=attribute_value,
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

        # ----------------------------------------------
        # COLOR
        # ----------------------------------------------

        if name in {
            "color",
            "colour",
        }:

            return (
                self._evaluate_color(
                    value=value,
                    label_tokens=
                        label_tokens,
                )
            )

        # ----------------------------------------------
        # GENERIC STRING ATTRIBUTE
        #
        # Example:
        #
        # size = large
        #
        # "large car" confirms it.
        # "car" does not.
        # ----------------------------------------------

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
                f"not encoded in track label"
            ),
        )

    def _evaluate_color(
        self,
        value,
        label_tokens: set[str],
    ) -> AttributeDecision:

        requested_colors = (
            self._value_tokens(
                value
            )
            &
            self.COLOR_WORDS
        )

        if not requested_colors:

            return AttributeDecision(
                status="UNVERIFIED",
                reason=(
                    f"unknown color: {value}"
                ),
            )

        stored_colors = (
            label_tokens
            &
            self.COLOR_WORDS
        )

        # ----------------------------------------------
        # Index explicitly says red, blue, etc.
        # ----------------------------------------------

        if (
            requested_colors
            &
            stored_colors
        ):

            return AttributeDecision(
                status="MATCH",
                reason=(
                    "color verified "
                    "from track label"
                ),
            )

        # ----------------------------------------------
        # Index explicitly contains another color.
        #
        # requested: red
        # track: blue car
        #
        # Definitely reject.
        # ----------------------------------------------

        if stored_colors:

            return AttributeDecision(
                status="REJECT",
                reason=(
                    "track has conflicting "
                    f"color: "
                    f"{', '.join(stored_colors)}"
                ),
            )

        # ----------------------------------------------
        # Generic "car" track.
        #
        # We cannot prove it is red.
        # ----------------------------------------------

        return AttributeDecision(
            status="UNVERIFIED",
            reason=(
                "track color is not "
                "currently verified"
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

        #
        # Support something like:
        #
        # ["red", "large"]
        #
        # temporarily.
        #

        if isinstance(
            attributes,
            list,
        ):

            return {
                f"attribute_{index}":
                    value

                for index, value
                in enumerate(attributes)
            }

        return {
            "attribute":
                attributes
        }

    def _tokens(
        self,
        text: str,
    ) -> set[str]:

        text = (
            str(text)
            .lower()
            .replace("-", " ")
            .replace("_", " ")
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

            tokens = set()

            for item in value:

                tokens.update(
                    self._tokens(
                        str(item)
                    )
                )

            return tokens

        return self._tokens(
            str(value)
        )