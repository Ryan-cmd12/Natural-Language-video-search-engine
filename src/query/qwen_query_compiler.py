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
        device,
        model_name: str = (
            "Qwen/"
            "Qwen2.5-3B-Instruct"
        ),
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

        prompt = """
    You are a compiler for a natural-language video search engine.

    Convert the user query into structured JSON.

    Return ONLY valid JSON.
    Do not answer the query.
    Do not invent objects, attributes, actions, relationships,
    or temporal constraints.

    Schema:

    {
        "target": "object | event | event_sequence | scene",

        "entities": [
            {
                "id": "entity_1",
                "concept": "human",
                "attributes": {}
            }
        ],

        "actions": [
            {
                "id": "action_1",
                "verb": "walk",
                "actor": "entity_1",
                "object": null,
                "target": null,
                "attributes": {}
            }
        ],

        "relationships": [],

        "temporal_constraints": []
    }

    ENTITY RULES:

    1. Every visible physical object, person, animal,
    or thing explicitly mentioned in the query MUST
    become an entity.

    2. Preserve the base noun used by the user whenever
    possible.

    Examples:

    "human" -> concept="human"
    "person" -> concept="person"
    "car" -> concept="car"
    "dog" -> concept="dog"

    Do NOT replace "human" with "car".
    Do NOT invent an entity that is not mentioned.

    3. Descriptive visual properties belong in attributes.

    "red car"
    -> concept="car"
    -> attributes={"color": "red"}

    "large dog"
    -> concept="dog"
    -> attributes={"size": "large"}

    4. Attributes MUST come only from words explicitly
    present in the query.

    5. If no attribute is stated, use {}.

    ACTION RULES:

    1. A verb describing what an entity is doing MUST
    become an action.

    2. Normalize verbs to their base form.

    walking -> walk
    running -> run
    sitting -> sit
    standing -> stand
    carrying -> carry
    holding -> hold
    driving -> drive
    eating -> eat

    3. The entity performing the action MUST be referenced
    through the action's "actor" field.

    4. Do not put actions inside entity attributes.

    5. If the query contains exactly one action:
    target = "event"

    6. If the query contains multiple ordered actions:
    target = "event_sequence"

    7. If the query contains no action and only asks for
    an object:
    target = "object"

    RELATIONSHIP RULES:

    Create relationships only when explicitly stated.

    Example:

    "human near car"

    entities:
    - human
    - car

    relationship:
    {
        "subject": "entity_1",
        "predicate": "near",
        "object": "entity_2"
    }

    TEMPORAL RULES:

    Create temporal constraints only when the query explicitly
    specifies ordering or duration.

    Examples:

    "human walks then sits"

    -> two actions
    -> target="event_sequence"
    -> temporal relation="then"

    EXAMPLES:

    Query:
    "red car"

    Output:
    {
        "target": "object",
        "entities": [
            {
                "id": "entity_1",
                "concept": "car",
                "attributes": {
                    "color": "red"
                }
            }
        ],
        "actions": [],
        "relationships": [],
        "temporal_constraints": []
    }

    Query:
    "human walking"

    Output:
    {
        "target": "event",
        "entities": [
            {
                "id": "entity_1",
                "concept": "human",
                "attributes": {}
            }
        ],
        "actions": [
            {
                "id": "action_1",
                "verb": "walk",
                "actor": "entity_1",
                "object": null,
                "target": null,
                "attributes": {}
            }
        ],
        "relationships": [],
        "temporal_constraints": []
    }

    Query:
    "dog running"

    Output:
    {
        "target": "event",
        "entities": [
            {
                "id": "entity_1",
                "concept": "dog",
                "attributes": {}
            }
        ],
        "actions": [
            {
                "id": "action_1",
                "verb": "run",
                "actor": "entity_1",
                "object": null,
                "target": null,
                "attributes": {}
            }
        ],
        "relationships": [],
        "temporal_constraints": []
    }

    FINAL CHECK:

    Before returning:

    1. Every explicitly mentioned physical entity is included.
    2. No unmentioned entity has been added.
    3. Every action explicitly mentioned is included.
    4. Action verbs are normalized to their base form.
    5. Every action references the correct entity.
    6. Attribute values come only from the query.
    7. Return valid JSON only.

    USER QUERY:
    "__USER_QUERY__"

    OUTPUT:
    """.strip()

        return prompt.replace(
            "__USER_QUERY__",
            query,
        )

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