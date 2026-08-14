import json

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.query.models import (
    EntitySpec,
    ActionSpec,
    RelationshipSpec,
    TemporalConstraint,
    CompiledQuery,
)


class QwenQueryCompiler:

    def __init__(
        self,
        model_name: str = (
            "Qwen/"
            "Qwen2.5-3B-Instruct"
        ),
        device: str = "cpu",
    ):

        self.model_name = model_name
        self.device = device

        print(
            f"Loading query compiler: "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_name
            )
        )

        if device == "cpu":

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,

                    torch_dtype=
                        torch.float32,

                    low_cpu_mem_usage=
                        True,
                )
                .to("cpu")
            )

        else:

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,

                    torch_dtype="auto",
                    device_map="auto",
                )
            )

        self.model.eval()

    # ======================================
    # Prompt
    # ======================================

    @staticmethod
    def _build_prompt(
        query: str,
    ) -> str:

        return f"""
You are a compiler for a natural-language video
search engine.

Convert the user's request into structured JSON.

Do NOT answer the user's question.
Do NOT invent events or objects.
Only represent what the query asks for.

User query:

"{query}"

Return ONLY valid JSON.

Schema:

{{
    "target": "object | event | event_sequence | scene",

    "entities": [
        {{
            "id": "entity_1",
            "concept": "person",
            "attributes": {{
                "shirt_color": "red"
            }}
        }}
    ],

    "actions": [
        {{
            "id": "action_1",
            "verb": "carry",
            "actor": "entity_1",
            "object": "entity_2",
            "target": null,
            "attributes": {{}}
        }}
    ],

    "relationships": [
        {{
            "subject": "entity_1",
            "predicate": "near",
            "object": "entity_2"
        }}
    ],

    "temporal_constraints": [
        {{
            "relation": "before | after | then | within | duration_gt | duration_lt",
            "first": "action_1",
            "second": "action_2",
            "value": null,
            "unit": null
        }}
    ]
}}

Rules:

1. Every entity gets a unique id.
2. Keep the concept short and object-oriented.
3. Put visual properties into attributes.
4. Actions must reference entity ids.
5. Do not create an action unless the query
   actually contains an action.
6. Use target="object" for simple queries like
   "red car".
7. Use target="event" for one action.
8. Use target="event_sequence" for multiple
   ordered actions.
9. Use temporal constraints only when the query
   actually specifies ordering or duration.
10. For phrases like "then", create a temporal
    constraint linking the actions.
11. null must be JSON null.
12. Output no markdown and no explanation.
13. If the query names a visible object, person,
    animal, or physical thing, it MUST appear in
    "entities".

14. A single object noun is still an entity.

15. Never return target="object" with an empty
    "entities" list when the query names an object.

Examples:

Query: "car"

{{
    "target": "object",
    "entities": [
        {{
            "id": "entity_1",
            "concept": "car",
            "attributes": {{}}
        }}
    ],
    "actions": [],
    "relationships": [],
    "temporal_constraints": []
}}

Query: "red car"

{{
    "target": "object",
    "entities": [
        {{
            "id": "entity_1",
            "concept": "car",
            "attributes": {{
                "color": "red"
            }}
        }}
    ],
    "actions": [],
    "relationships": [],
    "temporal_constraints": []
}}
""".strip()

    # ======================================
    # Extract JSON
    # ======================================

    @staticmethod
    def _extract_json(
        response: str,
    ) -> dict:

        cleaned = (
            response
            .strip()
        )

        if cleaned.startswith(
            "```"
        ):

            cleaned = (
                cleaned
                .replace(
                    "```json",
                    "",
                    1,
                )
                .replace(
                    "```",
                    "",
                )
                .strip()
            )

        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if (
            start == -1
            or end == -1
        ):

            raise ValueError(
                "Compiler did not "
                "return JSON."
            )

        cleaned = cleaned[
            start:end + 1
        ]

        return json.loads(
            cleaned
        )

    # ======================================
    # Validate references
    # ======================================

    @staticmethod
    def _validate(
        data: dict,
    ):

        entities = (
            data.get(
                "entities",
                [],
            )
        )

        actions = (
            data.get(
                "actions",
                [],
            )
        )

        entity_ids = {
            entity["id"]
            for entity
            in entities
        }

        action_ids = {
            action["id"]
            for action
            in actions
        }

        # ------------------------------
        # Check action references
        # ------------------------------

        for action in actions:

            for key in [
                "actor",
                "object",
                "target",
            ]:

                reference = (
                    action.get(
                        key
                    )
                )

                if reference is None:
                    continue

                if (
                    reference
                    not in entity_ids
                ):

                    raise ValueError(
                        f"Action "
                        f"{action['id']} "
                        f"references unknown "
                        f"entity: {reference}"
                    )

        # ------------------------------
        # Check relationships
        # ------------------------------

        for relationship in (
            data.get(
                "relationships",
                [],
            )
        ):

            subject = (
                relationship[
                    "subject"
                ]
            )

            obj = (
                relationship[
                    "object"
                ]
            )

            if (
                subject
                not in entity_ids
            ):

                raise ValueError(
                    f"Unknown relationship "
                    f"subject: {subject}"
                )

            if (
                obj
                not in entity_ids
            ):

                raise ValueError(
                    f"Unknown relationship "
                    f"object: {obj}"
                )

        # ------------------------------
        # Temporal relationships
        # ------------------------------

        for constraint in (
            data.get(
                "temporal_constraints",
                [],
            )
        ):

            first = (
                constraint.get(
                    "first"
                )
            )

            second = (
                constraint.get(
                    "second"
                )
            )

            if (
                first is not None
                and
                first not in action_ids
            ):

                raise ValueError(
                    f"Unknown temporal "
                    f"action: {first}"
                )

            if (
                second is not None
                and
                second not in action_ids
            ):

                raise ValueError(
                    f"Unknown temporal "
                    f"action: {second}"
                )

    # ======================================
    # Convert JSON → dataclasses
    # ======================================

    @staticmethod
    def _to_compiled_query(
        query: str,
        data: dict,
    ) -> CompiledQuery:

        entities = [
            EntitySpec(
                id=entity["id"],

                concept=
                    entity["concept"],

                attributes=
                    entity.get(
                        "attributes",
                        {},
                    ),
            )

            for entity
            in data.get(
                "entities",
                []
            )
        ]

        actions = [
            ActionSpec(
                id=action["id"],

                verb=
                    action["verb"],

                actor=
                    action.get(
                        "actor"
                    ),

                object=
                    action.get(
                        "object"
                    ),

                target=
                    action.get(
                        "target"
                    ),

                attributes=
                    action.get(
                        "attributes",
                        {},
                    ),
            )

            for action
            in data.get(
                "actions",
                []
            )
        ]

        relationships = [
            RelationshipSpec(
                subject=
                    relationship[
                        "subject"
                    ],

                predicate=
                    relationship[
                        "predicate"
                    ],

                object=
                    relationship[
                        "object"
                    ],
            )

            for relationship
            in data.get(
                "relationships",
                []
            )
        ]

        temporal_constraints = [
            TemporalConstraint(
                relation=
                    constraint[
                        "relation"
                    ],

                first=
                    constraint.get(
                        "first"
                    ),

                second=
                    constraint.get(
                        "second"
                    ),

                value=
                    constraint.get(
                        "value"
                    ),

                unit=
                    constraint.get(
                        "unit"
                    ),
            )

            for constraint
            in data.get(
                "temporal_constraints",
                []
            )
        ]

        return CompiledQuery(
            original_query=
                query,

            target=
                data.get(
                    "target",
                    "object",
                ),

            entities=
                entities,

            actions=
                actions,

            relationships=
                relationships,

            temporal_constraints=
                temporal_constraints,
        )

    # ======================================
    # Main compiler
    # ======================================

    def compile(
        self,
        query: str,
    ) -> CompiledQuery:

        prompt = (
            self._build_prompt(
                query
            )
        )

        messages = [
            {
                "role": "system",

                "content":
                    (
                        "You compile video "
                        "search queries into "
                        "strict JSON."
                    ),
            },

            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = (
            self.tokenizer
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=
                    True,
            )
        )

        inputs = (
            self.tokenizer(
                [text],

                return_tensors=
                    "pt",
            )
        )

        inputs = {
            key: value.to(
                self.model.device
            )

            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            generated = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=500,

                    do_sample=False,
                )
            )

        output_tokens = (
            generated[
                0,
                inputs[
                    "input_ids"
                ].shape[1]:
            ]
        )

        response = (
            self.tokenizer.decode(
                output_tokens,

                skip_special_tokens=
                    True,
            )
        )

        data = (
            self._extract_json(
                response
            )
        )

        self._validate(
            data
        )

        return (
            self._to_compiled_query(
                query=query,
                data=data,
            )
        )