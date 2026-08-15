from dataclasses import dataclass


@dataclass
class TemporalResult:

    status: str

    # MATCH
    # REJECT
    # UNVERIFIED

    confidence: float

    relation: str

    gap_seconds: float | None

    reason: str


class TemporalFilter:

    SUPPORTED_RELATIONS = {
        "before",
        "after",
        "overlaps",
        "during",
        "contains",
        "within",
    }

    def evaluate(
        self,
        first,
        second,
        relation: str,
        value: float | None = None,
        unit: str | None = None,
    ) -> TemporalResult:

        relation = (
            relation
            .strip()
            .lower()
        )

        if (
            relation
            not in
            self.SUPPORTED_RELATIONS
        ):
            return TemporalResult(
                status="UNVERIFIED",
                confidence=0.0,
                relation=relation,
                gap_seconds=None,
                reason=(
                    f"Unsupported temporal "
                    f"relation '{relation}'."
                ),
            )

        first_start = (
            first.start_time
        )

        first_end = (
            first.end_time
        )

        second_start = (
            second.start_time
        )

        second_end = (
            second.end_time
        )

        # ==========================================
        # BEFORE
        # ==========================================

        if relation == "before":

            gap = (
                second_start
                -
                first_end
            )

            match = gap >= 0

        # ==========================================
        # AFTER
        # ==========================================

        elif relation == "after":

            gap = (
                first_start
                -
                second_end
            )

            match = gap >= 0

        # ==========================================
        # OVERLAPS
        # ==========================================

        elif relation == "overlaps":

            overlap_start = max(
                first_start,
                second_start,
            )

            overlap_end = min(
                first_end,
                second_end,
            )

            overlap = (
                overlap_end
                -
                overlap_start
            )

            gap = None

            match = (
                overlap >= 0
            )

        # ==========================================
        # DURING
        # ==========================================

        elif relation == "during":

            gap = None

            match = (
                first_start
                >=
                second_start
                and
                first_end
                <=
                second_end
            )

        # ==========================================
        # CONTAINS
        # ==========================================

        elif relation == "contains":

            gap = None

            match = (
                first_start
                <=
                second_start
                and
                first_end
                >=
                second_end
            )

        # ==========================================
        # WITHIN
        # ==========================================

        else:

            threshold = (
                self._to_seconds(
                    value,
                    unit,
                )
            )

            if threshold is None:

                return TemporalResult(
                    status="UNVERIFIED",
                    confidence=0.0,
                    relation=relation,
                    gap_seconds=None,
                    reason=(
                        "'within' requires "
                        "a value and unit."
                    ),
                )

            gap = self._interval_gap(
                first_start,
                first_end,
                second_start,
                second_end,
            )

            match = (
                gap
                <=
                threshold
            )

        return TemporalResult(
            status=(
                "MATCH"
                if match
                else "REJECT"
            ),

            confidence=(
                1.0
                if match
                else 0.0
            ),

            relation=relation,

            gap_seconds=gap,

            reason=(
                f"Temporal relation "
                f"'{relation}' "
                f"{'matched' if match else 'did not match'}."
            ),
        )

    def _interval_gap(
        self,
        first_start,
        first_end,
        second_start,
        second_end,
    ):

        if (
            first_end
            <
            second_start
        ):

            return (
                second_start
                -
                first_end
            )

        if (
            second_end
            <
            first_start
        ):

            return (
                first_start
                -
                second_end
            )

        return 0.0

    def _to_seconds(
        self,
        value,
        unit,
    ):

        if value is None:
            return None

        unit = (
            unit
            or "seconds"
        ).lower()

        conversions = {
            "second": 1.0,
            "seconds": 1.0,
            "sec": 1.0,
            "s": 1.0,

            "minute": 60.0,
            "minutes": 60.0,
            "min": 60.0,
            "m": 60.0,

            "hour": 3600.0,
            "hours": 3600.0,
            "h": 3600.0,
        }

        multiplier = (
            conversions.get(
                unit
            )
        )

        if multiplier is None:
            return None

        return (
            float(value)
            *
            multiplier
        )