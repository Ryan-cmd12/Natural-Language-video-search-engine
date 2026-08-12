import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from qwen_vl_utils import process_vision_info

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass
class VerificationResult:
    match: bool
    confidence: float
    reason: str
    evidence_frames: list[int]
    raw_response: str


class QwenVideoVerifier:

    def __init__(
        self,
        max_frames,
        model_name: str = (
            "Qwen/Qwen2.5-VL-3B-Instruct"
        ),
        device: str = None,
    ):

        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_frames = max_frames

        print(
            f"Loading VLM verifier: "
            f"{model_name}"
            f"using {self.device}"
        )

        # Restrict image resolution a little.
        # We don't need full-resolution 1080p
        # frames just to check whether an object
        # or event exists.

        min_pixels = 224 * 224
        max_pixels = 512 * 512

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        )

        if device == "cpu":

            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            )

            self.model = self.model.to(
                "cpu"
            )

        else:

            self.model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                )
            )

        self.model.eval()

    # =========================================
    # Select a small number of frames
    # =========================================

    def _select_frames(
        self,
        frame_paths: list[str],
    ) -> list[str]:

        if not frame_paths:
            raise ValueError(
                "Segment contains no frames."
            )

        if (
            len(frame_paths)
            <= self.max_frames
        ):
            return frame_paths

        indices = np.linspace(
            0,
            len(frame_paths) - 1,
            num=self.max_frames,
            dtype=int,
        )

        return [
            frame_paths[index]
            for index in indices
        ]

    # =========================================
    # Convert Windows paths to file:// URIs
    # =========================================

    @staticmethod
    def _to_local_path(path: str,) -> str:

        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        return str(file_path)

    # =========================================
    # Verification prompt
    # =========================================

    @staticmethod
    def _build_prompt(
        query: str,
        start_time: float,
        end_time: float,
        frame_count: int,
    ) -> str:

        return f"""
        You are a strict video-search verification system.

        The user searched for:

        "{query}"

        You are given {frame_count} frames in chronological
        order from the video segment:

        {start_time:.2f}s -> {end_time:.2f}s

        Your task is to determine whether this video segment
        ACTUALLY contains visual evidence that satisfies the
        user's search query.

        Important rules:

        1. Do not assume a match exists.
        2. Similar-looking content is not enough.
        3. The requested object, person, attribute, action,
        or scene must be visually supported.
        4. If the query asks for an object such as "bus",
        the actual object must be visible.
        5. Text merely mentioning the object does NOT count
        unless the user's query specifically asks for
        visible text.
        6. Do not infer objects that are not visible.
        7. If evidence is weak or ambiguous, return false.
        8. For actions, use the chronological sequence of
        frames rather than judging from one frame alone.
        9. confidence means the strength of evidence that
        the QUERY IS ACTUALLY PRESENT.
        It is NOT confidence in your ability to answer.

        Return ONLY valid JSON with exactly these fields:

        {{
            "match": <true or false>,
            "confidence": <number between 0.0 and 1.0>,
            "reason": "<brief factual description of what is actually visible>",
            "evidence_frames": <list of 1-based frame numbers supporting the decision>
        }}

        Do not copy example descriptions.
        Describe only what is actually visible in the provided frames.

        Frame numbers are 1-based.
        Do not output markdown.
        Do not output anything outside the JSON.
        """.strip()

    # =========================================
    # Parse model JSON safely
    # =========================================

    @staticmethod
    def _parse_response(
        raw_response: str,
    ) -> VerificationResult:

        cleaned = (
            raw_response
            .strip()
        )

        # Remove markdown fences if
        # the model ignored instructions.

        if cleaned.startswith("```"):

            cleaned = cleaned.replace(
                "```json",
                "",
                1,
            )

            cleaned = cleaned.replace(
                "```",
                "",
            )

            cleaned = cleaned.strip()

        # Try to extract only the JSON
        # object if extra text slipped in.

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1:

            cleaned = cleaned[
                start:end + 1
            ]

        try:

            data = json.loads(
                cleaned
            )

        except json.JSONDecodeError:

            # Fail closed.
            #
            # If the verifier produces an
            # invalid response, we do NOT
            # trust the candidate.

            return VerificationResult(
                match=False,
                confidence=0.0,
                reason=(
                    "Verifier returned "
                    "invalid JSON."
                ),
                evidence_frames=[],
                raw_response=raw_response,
            )

        match = bool(
            data.get(
                "match",
                False,
            )
        )

        try:

            confidence = float(
                data.get(
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
                confidence,
                1.0,
            ),
        )

        reason = str(
            data.get(
                "reason",
                "",
            )
        )

        evidence_frames = (
            data.get(
                "evidence_frames",
                [],
            )
        )

        if not isinstance(
            evidence_frames,
            list,
        ):
            evidence_frames = []

        evidence_frames = [
            int(frame)
            for frame in evidence_frames
            if isinstance(
                frame,
                (int, float)
            )
        ]

        return VerificationResult(
            match=match,
            confidence=confidence,
            reason=reason,
            evidence_frames=
                evidence_frames,
            raw_response=
                raw_response,
        )

    # =========================================
    # Main verification function
    # =========================================

    def verify(
        self,
        query: str,
        segment: dict,
    ) -> VerificationResult:

        frame_paths = (
            segment[
                "frame_paths"
            ]
        )

        selected_frames = (
            self._select_frames(
                frame_paths
            )
        )

        frame_paths_for_qwen = [
            self._to_local_path(path)
            for path in selected_frames
        ]

        prompt = (
            self._build_prompt(
                query=query,
                start_time=
                    segment["start_time"],
                end_time=
                    segment["end_time"],
                frame_count=
                    len(selected_frames),
            )
        )

        # Qwen supports an ordered list
        # of images as video frames.

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frame_paths_for_qwen,
                        "fps": 1.0,
                    },
                    {
                        "type": "text",
                        "text":
                            prompt,
                    },
                ],
            }
        ]

        text = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        (
            image_inputs,
            video_inputs,
        ) = process_vision_info(
            messages
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.device == "cpu":

            inputs = inputs.to(
                "cpu"
            )

        else:

            inputs = inputs.to(
                self.model.device
            )

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=180,

                    do_sample=False,
                )
            )

        # Remove the prompt tokens so
        # we're only decoding the answer.

        generated_ids_trimmed = [
            output_ids[
                len(input_ids):
            ]

            for (
                input_ids,
                output_ids,
            ) in zip(
                inputs.input_ids,
                generated_ids,
            )
        ]

        output_text = (
            self.processor
            .batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

        print("\n[QWEN RAW RESPONSE]")
        print(output_text)
        print()

        return self._parse_response(
            output_text
        )