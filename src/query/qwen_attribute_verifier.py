import json
import re

from dataclasses import dataclass
from typing import Callable, Any

from src.models.tracked_object import (
    ObjectTrack,
)

from src.query.track_frame_sampler import (
    TrackFrameSampler,
)


@dataclass
class VisualAttributeResult:

    status: str

    # MATCH
    # REJECT
    # UNVERIFIED

    confidence: float

    reason: str


class QwenAttributeVerifier:

    def __init__(
        self,
        ask_fn: Callable[
            [str, str],
            Any,
        ],
        sampler: TrackFrameSampler | None = None,
    ):

        self.ask_fn = ask_fn

        self.sampler = (
            sampler
            or
            TrackFrameSampler()
        )

    # ==================================================
    # VERIFY
    # ==================================================

    def verify_track(
        self,
        video_path: str,
        video_id: str,
        track: ObjectTrack,
        entity_label: str,
        attributes: dict,
    ) -> VisualAttributeResult:

        image_path = (
            self.sampler.build_contact_sheet(
                video_path=video_path,
                video_id=video_id,
                track=track,
            )
        )

        if image_path is None:

            return VisualAttributeResult(
                status="UNVERIFIED",
                confidence=0.0,
                reason=(
                    "could not sample "
                    "track frames"
                ),
            )

        prompt = self._build_prompt(
            entity_label=
                entity_label,

            attributes=
                attributes,
        )

        raw = self.ask_fn(
            str(image_path),
            prompt,
        )

        payload = (
            self._parse_response(
                raw
            )
        )

        if payload is None:

            return VisualAttributeResult(
                status="UNVERIFIED",
                confidence=0.0,
                reason=(
                    "could not parse "
                    "Qwen response"
                ),
            )

        status = (
            str(
                payload.get(
                    "status",
                    "UNVERIFIED",
                )
            )
            .upper()
            .strip()
        )

        if status not in {
            "MATCH",
            "REJECT",
            "UNVERIFIED",
        }:

            status = (
                "UNVERIFIED"
            )

        try:

            confidence = float(
                payload.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return VisualAttributeResult(
            status=status,

            confidence=confidence,

            reason=str(
                payload.get(
                    "reason",
                    "",
                )
            ),
        )

    # ==================================================
    # PROMPT
    # ==================================================

    def _build_prompt(
        self,
        entity_label: str,
        attributes: dict,
    ) -> str:

        attributes_json = (
            json.dumps(
                attributes,
                ensure_ascii=False,
            )
        )

        return f"""
You are verifying visual attributes of ONE tracked object.

The image is a contact sheet containing several crops of the SAME tracked object at different moments in a video.

Entity type:
{entity_label}

Required attributes:
{attributes_json}

Determine whether the tracked object satisfies ALL requested attributes.

Rules:

1. Judge only the visible tracked object.
2. Do not infer an attribute if it cannot be seen clearly.
3. If the requested attribute clearly matches, return MATCH.
4. If it clearly conflicts, return REJECT.
5. If there is insufficient visual evidence, return UNVERIFIED.
6. Do not treat semantic similarity as evidence.
7. For colors, the requested color must actually be visually present on the object.

Return JSON only:

{{
    "status": "MATCH",
    "confidence": 0.0,
    "reason": "short explanation"
}}
""".strip()

    # ==================================================
    # RESPONSE
    # ==================================================

    def _parse_response(
        self,
        raw,
    ) -> dict | None:

        if isinstance(
            raw,
            dict,
        ):
            return raw

        text = str(raw).strip()

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if not match:
            return None

        try:

            return json.loads(
                match.group(0)
            )

        except json.JSONDecodeError:

            return None